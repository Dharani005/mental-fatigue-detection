"""
Comprehensive Desktop Tkinter Application for Mental Fatigue Detection & Data Collection.

Combines:
1. Live Arithmetic Typing Task with real-time multi-signal behavioral tracking
   (Keystrokes, Mouse motion/clicks, Response Times, Voluntary Breaks, Karolinska scale).
2. Desktop Fatigue Analytics & ML Prediction Explorer with embedded Matplotlib visualizations.
3. One-Click Power BI Data Export Center & schema guide.
"""

import os
import sys
import time
import random
import datetime
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# Matplotlib embedded in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import (
    discover_session_ids,
    extract_window_features,
    load_session_csvs,
    build_feature_table
)
from src.label_preparation import prepare_labels
from src.model_training import run_stage4_pipeline
from src.power_bi_export import export_power_bi_datasets


class MentalFatigueApp:
    """Main Tkinter Application Window containing the 3 core tabs."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mental Fatigue Detection System — Desktop Station")
        self.root.geometry("1100x780")
        self.root.minsize(980, 680)

        # Style configuration
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._configure_styles()

        self.data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Build Notebook / Tabbed Interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create Tab Frames
        self.tab_typing = ttk.Frame(self.notebook, padding=10)
        self.tab_analytics = ttk.Frame(self.notebook, padding=10)
        self.tab_powerbi = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_typing, text="  ⌨️ Typing & Cognitive Assessment  ")
        self.notebook.add(self.tab_analytics, text="  📊 Fatigue Analytics & ML Model  ")
        self.notebook.add(self.tab_powerbi, text="  📈 Power BI Export Center  ")

        # Initialize State for Typing Station
        self._init_typing_state()

        # Build UI for all tabs
        self._build_typing_tab()
        self._build_analytics_tab()
        self._build_powerbi_tab()

        # Bind tab change event to refresh analytics if needed
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _configure_styles(self):
        """Setup custom colors and fonts for widgets."""
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[12, 6])
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#0f172a")
        self.style.configure("Subheader.TLabel", font=("Segoe UI", 11), foreground="#475569")
        self.style.configure("Card.TFrame", background="#ffffff", relief="ridge", borderwidth=1)
        self.style.configure("MetricVal.TLabel", font=("Segoe UI", 18, "bold"), foreground="#1e40af", background="#ffffff")
        self.style.configure("MetricLbl.TLabel", font=("Segoe UI", 9, "bold"), foreground="#64748b", background="#ffffff")

    def _init_typing_state(self):
        """Initialize variables for active typing session."""
        self.session_id = None
        self.total_questions = 50
        self.current_q_idx = 0
        self.questions = []
        self.session_active = False
        self.is_on_break = False
        self.session_start_time = 0.0
        self.question_start_time = 0.0
        self.break_start_time = None
        self.last_mouse_move_time = 0.0
        self.mouse_sample_interval_sec = 0.05

        self.keystroke_logs = []
        self.mouse_event_logs = []
        self.performance_logs = []
        self.fatigue_rating_logs = []
        self.break_logs = []

        # Live stats
        self.correct_count = 0
        self.total_keypresses = 0

    # =========================================================================
    # TAB 1: TYPING & COGNITIVE ASSESSMENT STATION
    # =========================================================================
    def _build_typing_tab(self):
        """Construct Tab 1 UI."""
        # Top Config & Controls Bar
        top_bar = ttk.Frame(self.tab_typing)
        top_bar.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(top_bar, text="Cognitive Arithmetic Task & Typing Tracker", style="Header.TLabel").pack(side=tk.LEFT)

        self.btn_start_session = tk.Button(
            top_bar,
            text="▶ Start New Session (50 Qs)",
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            command=self._start_typing_session
        )
        self.btn_start_session.pack(side=tk.RIGHT, padx=5)

        self.btn_quick_test = tk.Button(
            top_bar,
            text="⚡ Quick Test (10 Qs)",
            bg="#0891b2",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=10,
            pady=6,
            command=lambda: self._start_typing_session(10)
        )
        self.btn_quick_test.pack(side=tk.RIGHT, padx=5)

        # Real-time KPI Metric Ribbon
        ribbon_frame = ttk.Frame(self.tab_typing)
        ribbon_frame.pack(fill=tk.X, pady=(0, 12))

        self.card_time = self._create_metric_card(ribbon_frame, "TIME ELAPSED", "00:00", 0)
        self.card_acc = self._create_metric_card(ribbon_frame, "ACCURACY", "100.0 %", 1)
        self.card_speed = self._create_metric_card(ribbon_frame, "TYPING SPEED", "0.00 KPS", 2)
        self.card_qprogress = self._create_metric_card(ribbon_frame, "PROGRESS", "0 / 50", 3)

        # Main Question Card Frame
        self.quiz_card = tk.Frame(self.tab_typing, bg="#ffffff", bd=2, relief="groove")
        self.quiz_card.pack(fill=tk.BOTH, expand=True, pady=5)

        self.lbl_quiz_status = tk.Label(
            self.quiz_card,
            text="Press 'Start New Session' to begin the multi-signal typing task.",
            font=("Segoe UI", 12),
            fg="#64748b",
            bg="#ffffff"
        )
        self.lbl_quiz_status.pack(pady=(20, 10))

        self.progress_bar = ttk.Progressbar(self.quiz_card, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=80, pady=(0, 20))

        self.lbl_question_text = tk.Label(
            self.quiz_card,
            text="--",
            font=("Segoe UI", 36, "bold"),
            fg="#0f172a",
            bg="#ffffff"
        )
        self.lbl_question_text.pack(pady=15)

        # Input Row
        entry_row = tk.Frame(self.quiz_card, bg="#ffffff")
        entry_row.pack(pady=15)

        tk.Label(entry_row, text="Answer:", font=("Segoe UI", 16, "bold"), fg="#334155", bg="#ffffff").pack(side=tk.LEFT, padx=10)

        self.entry_answer = tk.Entry(
            entry_row,
            font=("Segoe UI", 20, "bold"),
            width=10,
            justify="center",
            bd=2,
            relief="solid",
            state=tk.DISABLED
        )
        self.entry_answer.pack(side=tk.LEFT, padx=10)

        self.btn_submit = tk.Button(
            entry_row,
            text="Submit ⏎",
            font=("Segoe UI", 12, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=4,
            state=tk.DISABLED,
            command=self._handle_answer_submit
        )
        self.btn_submit.pack(side=tk.LEFT, padx=10)

        # Bottom Control & Feedback Strip
        bottom_bar = tk.Frame(self.quiz_card, bg="#f8fafc", height=50)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=0)

        self.btn_break = tk.Button(
            bottom_bar,
            text="☕ Take a Break",
            font=("Segoe UI", 10, "bold"),
            bg="#f59e0b",
            fg="#ffffff",
            activebackground="#d97706",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=4,
            state=tk.DISABLED,
            command=self._toggle_break
        )
        self.btn_break.pack(side=tk.LEFT, padx=20, pady=10)

        self.lbl_feedback = tk.Label(
            bottom_bar,
            text="Multi-signal logger ready.",
            font=("Segoe UI", 10, "italic"),
            fg="#64748b",
            bg="#f8fafc"
        )
        self.lbl_feedback.pack(side=tk.RIGHT, padx=20, pady=10)

        # Bind event listeners
        self.entry_answer.bind("<KeyPress>", lambda e: self._log_keystroke(e, "press"))
        self.entry_answer.bind("<KeyRelease>", lambda e: self._log_keystroke(e, "release"))
        self.entry_answer.bind("<Return>", lambda e: self._handle_answer_submit())
        self.root.bind("<Motion>", self._log_mouse_motion)
        self.root.bind("<Button-1>", self._log_mouse_click)

    def _create_metric_card(self, parent, title: str, init_val: str, col_idx: int) -> dict:
        """Helper to create a dashboard KPI metric card."""
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief="ridge", padx=15, pady=8)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        lbl_title = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg="#64748b", bg="#ffffff")
        lbl_title.pack(anchor=tk.W)

        lbl_val = tk.Label(card, text=init_val, font=("Segoe UI", 16, "bold"), fg="#1e40af", bg="#ffffff")
        lbl_val.pack(anchor=tk.W)

        return {"frame": card, "title": lbl_title, "val": lbl_val}

    def _start_typing_session(self, total_q: int = 50):
        """Start a new arithmetic typing session."""
        self.total_questions = total_q
        self.current_q_idx = 0
        self.correct_count = 0
        self.total_keypresses = 0
        self.is_on_break = False
        self.session_active = True
        self.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.keystroke_logs.clear()
        self.mouse_event_logs.clear()
        self.performance_logs.clear()
        self.fatigue_rating_logs.clear()
        self.break_logs.clear()

        # Generate math questions
        random.seed(int(time.time()))
        self.questions = []
        operators = ['+', '-', '*']
        for i in range(self.total_questions):
            op = random.choice(operators)
            if op == '+':
                n1, n2 = random.randint(15, 85), random.randint(15, 85)
                ans = n1 + n2
            elif op == '-':
                n1 = random.randint(30, 99)
                n2 = random.randint(10, n1)
                ans = n1 - n2
            else:
                n1, n2 = random.randint(4, 15), random.randint(4, 12)
                ans = n1 * n2
            self.questions.append({"q_num": i + 1, "text": f"{n1}  {op}  {n2} = ?", "answer": ans})

        self.session_start_time = time.time()
        self.question_start_time = time.time()

        # Enable controls
        self.entry_answer.config(state=tk.NORMAL)
        self.btn_submit.config(state=tk.NORMAL)
        self.btn_break.config(state=tk.NORMAL)
        self.btn_start_session.config(state=tk.DISABLED)
        self.btn_quick_test.config(state=tk.DISABLED)
        self.progress_bar.config(maximum=self.total_questions, value=0)

        self._load_current_question()
        self._update_live_timer()

    def _load_current_question(self):
        """Render question on screen and prepare input."""
        q_info = self.questions[self.current_q_idx]
        self.lbl_question_text.config(text=q_info["text"])
        self.lbl_quiz_status.config(text=f"Question {self.current_q_idx + 1} of {self.total_questions} — Type your answer and press Enter")
        self.card_qprogress["val"].config(text=f"{self.current_q_idx + 1} / {self.total_questions}")
        self.progress_bar.config(value=self.current_q_idx)
        self.entry_answer.delete(0, tk.END)
        self.entry_answer.focus_set()
        self.question_start_time = time.time()

    def _update_live_timer(self):
        """Update live elapsed time clock and typing metrics."""
        if not self.session_active:
            return

        elapsed = time.time() - self.session_start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.card_time["val"].config(text=f"{mins:02d}:{secs:02d}")

        if elapsed > 0:
            kps = self.total_keypresses / elapsed
            self.card_speed["val"].config(text=f"{kps:.2f} KPS")

        self.root.after(1000, self._update_live_timer)

    def _log_keystroke(self, event: tk.Event, event_type: str):
        """Log key press and release timing."""
        if not self.session_active or self.is_on_break:
            return

        curr_time = time.time()
        self.total_keypresses += 1
        key_symbol = event.keysym

        self.keystroke_logs.append({
            "session_id": self.session_id,
            "question_index": self.current_q_idx + 1,
            "timestamp": curr_time,
            "datetime_iso": datetime.datetime.fromtimestamp(curr_time).isoformat(),
            "event_type": event_type,
            "key_symbol": key_symbol,
            "is_backspace": (key_symbol == "BackSpace")
        })

    def _log_mouse_motion(self, event: tk.Event):
        """Log mouse movement with 50ms rate limit."""
        if not self.session_active or self.is_on_break:
            return

        curr_time = time.time()
        if curr_time - self.last_mouse_move_time < self.mouse_sample_interval_sec:
            return

        self.last_mouse_move_time = curr_time
        self.mouse_event_logs.append({
            "session_id": self.session_id,
            "question_index": self.current_q_idx + 1,
            "timestamp": curr_time,
            "event_type": "move",
            "x": event.x,
            "y": event.y,
            "target_element": str(event.widget)
        })

    def _log_mouse_click(self, event: tk.Event):
        """Log mouse clicks."""
        if not self.session_active or self.is_on_break:
            return

        curr_time = time.time()
        self.mouse_event_logs.append({
            "session_id": self.session_id,
            "question_index": self.current_q_idx + 1,
            "timestamp": curr_time,
            "event_type": "click",
            "x": event.x,
            "y": event.y,
            "target_element": str(event.widget)
        })

    def _toggle_break(self):
        """Handle taking a pause / resuming quiz."""
        if not self.session_active:
            return

        curr_time = time.time()
        if not self.is_on_break:
            self.is_on_break = True
            self.break_start_time = curr_time
            self.btn_break.config(text="▶ Resume Quiz", bg="#10b981", activebackground="#059669")
            self.entry_answer.config(state=tk.DISABLED)
            self.btn_submit.config(state=tk.DISABLED)
            self.lbl_feedback.config(text="Session paused. Take a deep breath!", fg="#d97706")
        else:
            self.is_on_break = False
            duration = curr_time - self.break_start_time
            self.break_logs.append({
                "session_id": self.session_id,
                "question_index": self.current_q_idx + 1,
                "break_start_time": self.break_start_time,
                "break_end_time": curr_time,
                "duration_sec": duration
            })
            self.break_start_time = None
            self.btn_break.config(text="☕ Take a Break", bg="#f59e0b", activebackground="#d97706")
            self.entry_answer.config(state=tk.NORMAL)
            self.btn_submit.config(state=tk.NORMAL)
            self.lbl_feedback.config(text="Session resumed.", fg="#059669")
            self.entry_answer.focus_set()

    def _handle_answer_submit(self):
        """Validate, evaluate answer, and record performance."""
        if not self.session_active or self.is_on_break:
            return

        user_raw = self.entry_answer.get().strip()
        if not user_raw:
            return

        curr_time = time.time()
        rt = curr_time - self.question_start_time
        elapsed_total = curr_time - self.session_start_time

        q_info = self.questions[self.current_q_idx]
        correct_ans = q_info["answer"]

        try:
            user_ans = int(user_raw)
            is_correct = 1 if (user_ans == correct_ans) else 0
        except ValueError:
            user_ans = user_raw
            is_correct = 0

        if is_correct:
            self.correct_count += 1
            self.lbl_feedback.config(text=f"✓ Correct! ({rt:.2f}s)", fg="#16a34a")
        else:
            self.lbl_feedback.config(text=f"✗ Answer was {correct_ans} ({rt:.2f}s)", fg="#dc2626")

        self.performance_logs.append({
            "session_id": self.session_id,
            "question_index": q_info["q_num"],
            "question_text": q_info["text"],
            "correct_answer": correct_ans,
            "user_answer": user_ans,
            "is_correct": is_correct,
            "response_time_sec": rt,
            "elapsed_session_time_sec": elapsed_total,
            "difficulty_tier": "Tier 2 - Medium Arithmetic"
        })

        acc_pct = (self.correct_count / (self.current_q_idx + 1)) * 100.0
        self.card_acc["val"].config(text=f"{acc_pct:.1f} %")

        self.current_q_idx += 1

        # Checkpoint prompt every 10 questions or on final question
        if self.current_q_idx % 10 == 0:
            self._prompt_karolinska_modal()

        if self.current_q_idx >= self.total_questions:
            self._complete_typing_session()
        else:
            self._load_current_question()

    def _prompt_karolinska_modal(self):
        """Show Karolinska Sleepiness Scale modal dialog."""
        modal = tk.Toplevel(self.root)
        modal.title("Fatigue Assessment Checkpoint")
        modal.geometry("480x420")
        modal.grab_set()
        modal.resizable(False, False)

        tk.Label(
            modal,
            text=f"Checkpoint: Question {self.current_q_idx} / {self.total_questions}",
            font=("Segoe UI", 14, "bold"),
            fg="#1e293b"
        ).pack(pady=(15, 6))

        tk.Label(
            modal,
            text="Please rate your current level of mental fatigue (1–7):",
            font=("Segoe UI", 11),
            fg="#475569"
        ).pack(pady=(0, 10))

        selected_rating = tk.IntVar(value=4)

        ratings_desc = [
            (1, "1 — Extremely Alert / Fully Awake"),
            (2, "2 — Very Alert"),
            (3, "3 — Alert / Normal State"),
            (4, "4 — Fairly Alert / Slight Decline"),
            (5, "5 — Neither Alert nor Sleepy"),
            (6, "6 — Noticeable Fatigue / Slower Reactions"),
            (7, "7 — Extremely Fatigued / Straining to Focus")
        ]

        frame_radios = tk.Frame(modal, bg="#f8fafc", bd=1, relief="solid", padx=15, pady=10)
        frame_radios.pack(fill=tk.BOTH, expand=True, padx=25, pady=5)

        for val, desc in ratings_desc:
            tk.Radiobutton(
                frame_radios,
                text=desc,
                variable=selected_rating,
                value=val,
                font=("Segoe UI", 10),
                bg="#f8fafc",
                anchor=tk.W
            ).pack(fill=tk.X, pady=2)

        def save_and_close():
            score = selected_rating.get()
            curr_t = time.time()
            self.fatigue_rating_logs.append({
                "session_id": self.session_id,
                "question_checkpoint": self.current_q_idx,
                "timestamp": curr_t,
                "datetime_iso": datetime.datetime.fromtimestamp(curr_t).isoformat(),
                "fatigue_score": score
            })
            modal.destroy()

        btn_confirm = tk.Button(
            modal,
            text="Confirm Rating & Continue",
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            relief="flat",
            padx=16,
            pady=6,
            command=save_and_close
        )
        btn_confirm.pack(pady=15)

        self.root.wait_window(modal)

    def _complete_typing_session(self):
        """Save CSV logs and finalize session."""
        self.session_active = False
        self.progress_bar.config(value=self.total_questions)
        self.entry_answer.config(state=tk.DISABLED)
        self.btn_submit.config(state=tk.DISABLED)
        self.btn_break.config(state=tk.DISABLED)
        self.btn_start_session.config(state=tk.NORMAL)
        self.btn_quick_test.config(state=tk.NORMAL)

        # Write files with schema guarantees
        schemas = {
            f"keystrokes_{self.session_id}.csv": (self.keystroke_logs, ["session_id", "question_index", "timestamp", "datetime_iso", "event_type", "key_symbol", "is_backspace"]),
            f"mouse_events_{self.session_id}.csv": (self.mouse_event_logs, ["session_id", "question_index", "timestamp", "event_type", "x", "y", "target_element"]),
            f"performance_{self.session_id}.csv": (self.performance_logs, ["session_id", "question_index", "question_text", "correct_answer", "user_answer", "is_correct", "response_time_sec", "elapsed_session_time_sec", "difficulty_tier"]),
            f"fatigue_ratings_{self.session_id}.csv": (self.fatigue_rating_logs, ["session_id", "question_checkpoint", "timestamp", "datetime_iso", "fatigue_score"]),
            f"breaks_{self.session_id}.csv": (self.break_logs, ["session_id", "question_index", "break_start_time", "break_end_time", "duration_sec"]),
        }

        for fname, (logs, cols) in schemas.items():
            fpath = os.path.join(self.data_dir, fname)
            df_out = pd.DataFrame(logs) if logs else pd.DataFrame(columns=cols)
            df_out.to_csv(fpath, index=False)

        messagebox.showinfo(
            "Session Completed! 🎉",
            f"Session {self.session_id} finished successfully!\n"
            f"• Accuracy: {(self.correct_count / self.total_questions) * 100:.1f}%\n"
            f"• Logs saved to 'data/' folder.\n\n"
            "Switch to the 'Fatigue Analytics' tab to view ML predictions and charts!"
        )
        self._refresh_analytics_dropdown()

    # =========================================================================
    # TAB 2: FATIGUE ANALYTICS & ML PREDICTION EXPLORER
    # =========================================================================
    def _build_analytics_tab(self):
        """Construct Tab 2 Desktop Analytics UI."""
        top_ctrl = ttk.Frame(self.tab_analytics)
        top_ctrl.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(top_ctrl, text="Session Analytics & XGBoost Fatigue Predictor", style="Header.TLabel").pack(side=tk.LEFT)

        self.btn_train_pipeline = tk.Button(
            top_ctrl,
            text="⚡ Run Full ML Pipeline",
            bg="#7c3aed",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            command=self._run_ml_pipeline_async
        )
        self.btn_train_pipeline.pack(side=tk.RIGHT, padx=5)

        self.combo_sessions = ttk.Combobox(top_ctrl, state="readonly", width=28, font=("Segoe UI", 10))
        self.combo_sessions.pack(side=tk.RIGHT, padx=10)
        self.combo_sessions.bind("<<ComboboxSelected>>", self._on_session_selected)

        ttk.Label(top_ctrl, text="Select Session:", font=("Segoe UI", 10, "bold")).pack(side=tk.RIGHT, padx=5)

        # KPI Metrics Cards Frame
        self.analytics_metrics_frame = ttk.Frame(self.tab_analytics)
        self.analytics_metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.ana_card_actual = self._create_metric_card(self.analytics_metrics_frame, "ACTUAL FATIGUE (1-7)", "--", 0)
        self.ana_card_pred = self._create_metric_card(self.analytics_metrics_frame, "PREDICTED FATIGUE", "--", 1)
        self.ana_card_rt = self._create_metric_card(self.analytics_metrics_frame, "AVG RESPONSE TIME", "--", 2)
        self.ana_card_careless = self._create_metric_card(self.analytics_metrics_frame, "CARELESS ERRORS", "--", 3)
        self.ana_card_breaks = self._create_metric_card(self.analytics_metrics_frame, "BREAKS TAKEN", "--", 4)

        # Matplotlib Canvas Frame for Analytics Visualizations
        self.chart_frame = tk.Frame(self.tab_analytics, bg="#ffffff", bd=1, relief="ridge")
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 4.5), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._refresh_analytics_dropdown()

    def _refresh_analytics_dropdown(self):
        """Populate session selection combobox."""
        s_ids = discover_session_ids(self.data_dir)
        if s_ids:
            self.combo_sessions["values"] = s_ids
            self.combo_sessions.current(len(s_ids) - 1)
            self._update_analytics_view(s_ids[-1])
        else:
            self.combo_sessions["values"] = ["No sessions found"]
            self.combo_sessions.current(0)

    def _on_session_selected(self, event=None):
        """Combobox selection handler."""
        selected = self.combo_sessions.get()
        if selected and selected != "No sessions found":
            self._update_analytics_view(selected)

    def _update_analytics_view(self, session_id: str):
        """Compute metrics and render charts for selected session."""
        df_win = extract_window_features(session_id, self.data_dir)
        if df_win.empty:
            return

        # Run or load model
        try:
            pipeline_res = run_stage4_pipeline(self.data_dir)
            model = (
                pipeline_res["clf_results"]["full_model"]
                if pipeline_res["is_classification"]
                else pipeline_res["reg_results"]["full_model"]
            )
            feature_names = pipeline_res["feature_names"]
            X_sess = df_win[feature_names]
            y_preds = model.predict(X_sess)
            shap_vals = pipeline_res["shap_values"]
        except Exception as e:
            print(f"[ANALYTICS ERROR] {e}")
            return

        y_actual = df_win["fatigue_score"].values
        checkpoints = df_win["window_checkpoint"].values

        # Update KPI Cards
        self.ana_card_actual["val"].config(text=f"{y_actual[-1]} / 7")
        self.ana_card_pred["val"].config(text=f"{y_preds[-1]:.2f}")
        self.ana_card_rt["val"].config(text=f"{df_win['mean_response_time'].mean():.2f} s")
        self.ana_card_careless["val"].config(text=f"{df_win['careless_error_rate'].mean() * 100:.1f} %")
        self.ana_card_breaks["val"].config(text=f"{int(df_win['break_count'].sum())} ({df_win['total_break_duration'].sum():.1f}s)")

        # Render 3 Subplots in Figure
        self.fig.clf()
        (ax1, ax2, ax3) = self.fig.subplots(1, 3)
        self.fig.subplots_adjust(wspace=0.35, left=0.06, right=0.96, top=0.88, bottom=0.15)

        # Plot 1: Predicted vs Actual Fatigue Trend
        ax1.plot(checkpoints, y_actual, marker="o", color="#ef4444", linewidth=2.2, label="Actual Karolinska")
        ax1.plot(checkpoints, y_preds, marker="s", linestyle="--", color="#3b82f6", linewidth=2.2, label="Predicted (XGB)")
        ax1.set_title("Fatigue Score vs Checkpoint", fontsize=11, fontweight="bold", color="#1e293b")
        ax1.set_xlabel("Question Number", fontsize=9)
        ax1.set_ylabel("Fatigue Score (1–7)", fontsize=9)
        ax1.set_ylim(0.5, 7.5)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=8)

        # Plot 2: Key Behavioral Degradation Signals
        press_norm = df_win["mean_press_dur"] / df_win["mean_press_dur"].max()
        rt_norm = df_win["mean_response_time"] / df_win["mean_response_time"].max()
        mouse_norm = df_win["mean_mouse_speed"] / df_win["mean_mouse_speed"].max()

        ax2.plot(checkpoints, press_norm, marker="^", color="#8b5cf6", label="Key Hold Time")
        ax2.plot(checkpoints, rt_norm, marker="v", color="#f59e0b", label="Response Time")
        ax2.plot(checkpoints, mouse_norm, marker="x", color="#10b981", label="Mouse Speed")
        ax2.set_title("Multi-Signal Normalized Shift", fontsize=11, fontweight="bold", color="#1e293b")
        ax2.set_xlabel("Question Number", fontsize=9)
        ax2.set_ylabel("Normalized Scale (0-1)", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best", fontsize=8)

        # Plot 3: Top SHAP Feature Importance Drivers
        shap_means = np.abs(shap_vals).mean(axis=0)
        df_shap = pd.DataFrame({"feature": feature_names, "impact": shap_means}).sort_values("impact").tail(6)
        y_pos = np.arange(len(df_shap))
        ax3.barh(y_pos, df_shap["impact"], color="#2563eb", alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(df_shap["feature"], fontsize=8)
        ax3.set_xlabel("Mean |SHAP Value|", fontsize=9)
        ax3.set_title("Global Top Fatigue Drivers", fontsize=11, fontweight="bold", color="#1e293b")
        ax3.grid(True, linestyle=":", alpha=0.6, axis="x")

        self.canvas.draw()

    def _run_ml_pipeline_async(self):
        """Execute full model training asynchronously."""
        def task():
            try:
                res = run_stage4_pipeline(self.data_dir)
                r2 = res["reg_results"]["mean_r2"]
                rmse = res["reg_results"]["mean_rmse"]
                self.root.after(0, lambda: messagebox.showinfo(
                    "ML Pipeline Completed",
                    f"Model Retrained Successfully (5-Fold Stratified CV):\n"
                    f"• Strategy: {res['chosen_strategy']}\n"
                    f"• Out-of-Fold R²: {r2:.4f}\n"
                    f"• Out-of-Fold RMSE: {rmse:.4f}\n"
                    f"• SHAP summary plot updated!"
                ))
                self.root.after(0, self._on_session_selected)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Pipeline Error", str(e)))

        threading.Thread(target=task, daemon=True).start()

    # =========================================================================
    # TAB 3: POWER BI EXPORT CENTER & SCHEMA GUIDE
    # =========================================================================
    def _build_powerbi_tab(self):
        """Construct Tab 3 Power BI Exporter & Documentation UI."""
        header_frame = ttk.Frame(self.tab_powerbi)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="Power BI Data Export & Modeling Center", style="Header.TLabel").pack(side=tk.LEFT)

        btn_export = tk.Button(
            header_frame,
            text="🚀 Export All Datasets for Power BI",
            bg="#0284c7",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=14,
            pady=6,
            command=self._trigger_powerbi_export
        )
        btn_export.pack(side=tk.RIGHT, padx=5)

        btn_open_folder = tk.Button(
            header_frame,
            text="📂 Open Export Folder",
            bg="#475569",
            fg="#ffffff",
            font=("Segoe UI", 10),
            relief="flat",
            padx=10,
            pady=6,
            command=self._open_export_directory
        )
        btn_open_folder.pack(side=tk.RIGHT, padx=5)

        # Status & Export Result Box
        self.lbl_pbi_status = tk.Label(
            self.tab_powerbi,
            text="Click 'Export All Datasets for Power BI' to generate normalized CSVs for Power BI Desktop.",
            font=("Segoe UI", 10, "italic"),
            fg="#0369a1",
            bg="#e0f2fe",
            padx=12,
            pady=8,
            anchor=tk.W
        )
        self.lbl_pbi_status.pack(fill=tk.X, pady=(0, 12))

        # Power BI Star Schema & Table Inventory
        notebook_docs = ttk.Notebook(self.tab_powerbi)
        notebook_docs.pack(fill=tk.BOTH, expand=True)

        frame_tables = ttk.Frame(notebook_docs, padding=10)
        frame_guide = ttk.Frame(notebook_docs, padding=10)

        notebook_docs.add(frame_tables, text=" 🗂️ Power BI Datasets Inventory ")
        notebook_docs.add(frame_guide, text=" 📖 Power BI Step-by-Step Setup Guide ")

        # Table Treeview in Frame Tables
        columns = ("table_name", "type", "key", "description")
        self.tree_tables = ttk.Treeview(frame_tables, columns=columns, show="headings", height=8)
        self.tree_tables.heading("table_name", text="Exported CSV File")
        self.tree_tables.heading("type", text="Power BI Role")
        self.tree_tables.heading("key", text="Primary / Join Key")
        self.tree_tables.heading("description", text="Contents & Analytics Use Case")

        self.tree_tables.column("table_name", width=220)
        self.tree_tables.column("type", width=120)
        self.tree_tables.column("key", width=160)
        self.tree_tables.column("description", width=420)

        datasets_meta = [
            ("power_bi_sessions_summary.csv", "Dimension Table", "session_id", "Session level totals: duration, accuracy, total breaks, fatigue change"),
            ("power_bi_window_features.csv", "Fact Table", "session_id + checkpoint", "10-Q window multi-signal features, actual Karolinska & XGBoost predictions"),
            ("power_bi_question_performance.csv", "Fact Table", "session_id + question_index", "Granular per-question math tasks, RT, errors (careless vs effortful)"),
            ("power_bi_feature_importance.csv", "Summary Table", "feature", "SHAP attribution & ANOVA F-scores for driver rankings in Power BI visuals")
        ]

        for row in datasets_meta:
            self.tree_tables.insert("", tk.END, values=row)

        self.tree_tables.pack(fill=tk.BOTH, expand=True)

        # Setup Guide Text in Frame Guide
        guide_text = (
            "HOW TO BUILD YOUR POWER BI DASHBOARD:\n"
            "--------------------------------------------------------------------------\n"
            "1. Open Microsoft Power BI Desktop.\n"
            "2. Click 'Get Data' -> 'Text/CSV' and import all 4 CSV files from 'data/power_bi_export/'.\n"
            "3. In the Model View (Star Schema):\n"
            "   - Connect `power_bi_sessions_summary[session_id]` (1) -> `power_bi_window_features[session_id]` (*)\n"
            "   - Connect `power_bi_sessions_summary[session_id]` (1) -> `power_bi_question_performance[session_id]` (*)\n"
            "4. Recommended Visualizations:\n"
            "   - Card 1: Average Actual Fatigue (`AVERAGE(fatigue_score)`)\n"
            "   - Card 2: Average Predicted Fatigue (`AVERAGE(predicted_fatigue_score)`)\n"
            "   - Line Chart: Checkpoint vs Actual & Predicted Fatigue over Time\n"
            "   - Bar Chart: Feature Importance (`mean_abs_shap_impact` by `feature`)\n"
            "   - Donut Chart: Error Breakdown (Careless vs Effortful Errors)\n"
            "   - Scatter Plot: Response Time vs Keystroke Press Duration colored by Fatigue Category\n"
            "--------------------------------------------------------------------------\n"
            "Full DAX formulas and color schemes are detailed in: power_bi_dashboard_guide.md"
        )
        txt = tk.Text(frame_guide, font=("Consolas", 10), bg="#f8fafc", fg="#1e293b", wrap=tk.WORD)
        txt.insert(tk.END, guide_text)
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True)

    def _trigger_powerbi_export(self):
        """Export CSVs for Power BI."""
        try:
            export_info = export_power_bi_datasets(self.data_dir)
            if export_info:
                exp_path = export_info["export_dir"]
                self.lbl_pbi_status.config(
                    text=f"✓ Successfully exported 4 Power BI datasets ({export_info['total_sessions']} sessions, {export_info['total_windows']} window rows) to:\n{exp_path}",
                    fg="#15803d",
                    bg="#dcfce7"
                )
                messagebox.showinfo(
                    "Power BI Export Completed",
                    f"4 Clean Power BI CSV files generated successfully in:\n{exp_path}\n\n"
                    "You can now import them directly into Power BI Desktop!"
                )
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def _open_export_directory(self):
        """Open export directory in Windows File Explorer."""
        p = os.path.join(self.data_dir, "power_bi_export")
        os.makedirs(p, exist_ok=True)
        try:
            os.startfile(p)
        except Exception as e:
            messagebox.showinfo("Export Folder", f"Path:\n{p}")

    def _on_tab_changed(self, event):
        """Handle tab switch events."""
        selected_tab = self.notebook.index(self.notebook.select())
        if selected_tab == 1:  # Analytics tab
            self._on_session_selected()


def main():
    root = tk.Tk()
    app = MentalFatigueApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
