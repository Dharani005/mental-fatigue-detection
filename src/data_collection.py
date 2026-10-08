"""
Stage 1: Mental Fatigue Data Collection Application.

This module implements a desktop GUI using Tkinter for collecting multi-signal
behavioral, temporal, mouse, keystroke, and self-reported mental fatigue data
during a timed arithmetic task.

Outputs 5 distinct CSV logs per session tagged by a unique session ID.
"""

import os
import sys
import time
import random
import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


class FatigueDataCollectorApp:
    """
    Tkinter Application for mental fatigue data collection across 50 math questions.
    """

    def __init__(self, root: tk.Tk, total_questions: int = 50, difficulty_tier: str = "Tier 2 - Medium Arithmetic"):
        """
        Initialize application window, variables, state, and event bindings.

        Args:
            root: Tkinter root window instance.
            total_questions: Total number of arithmetic questions (default 50).
            difficulty_tier: Fixed difficulty label for analysis control.
        """
        self.root = root
        self.root.title("Mental Fatigue Study — Timed Arithmetic Task")
        self.root.geometry("700x550")
        self.root.resizable(False, False)

        # Configuration & Parameters
        self.total_questions = total_questions
        self.difficulty_tier = difficulty_tier
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = DEFAULT_DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # State tracking
        self.current_q_idx = 0
        self.session_start_time = time.time()
        self.question_start_time = time.time()
        self.is_on_break = False
        self.break_start_time = None
        self.last_mouse_move_time = 0.0
        self.mouse_sample_interval_sec = 0.05  # 50ms sampling limit for mouse motion

        # Data logs
        self.keystroke_logs = []
        self.mouse_event_logs = []
        self.performance_logs = []
        self.fatigue_rating_logs = []
        self.break_logs = []

        # Generate math questions
        self.questions = self._generate_questions(self.total_questions)

        # Build UI layout
        self._build_ui()

        # Bind event listeners
        self._bind_events()

        # Start first question
        self._load_next_question()

    def _generate_questions(self, count: int) -> list:
        """
        Generate a set of arithmetic questions at a fixed difficulty tier.

        Args:
            count: Number of questions to generate.

        Returns:
            List of dicts containing question text and correct integer answer.
        """
        random.seed(42 + count)  # Consistent questions for benchmark reproducibility
        questions = []
        operators = ['+', '-', '*']

        for i in range(count):
            op = random.choice(operators)
            if op == '+':
                num1 = random.randint(15, 85)
                num2 = random.randint(15, 85)
                ans = num1 + num2
            elif op == '-':
                num1 = random.randint(30, 99)
                num2 = random.randint(10, num1)
                ans = num1 - num2
            else:  # '*'
                num1 = random.randint(4, 15)
                num2 = random.randint(4, 12)
                ans = num1 * num2

            questions.append({
                "q_num": i + 1,
                "text": f"{num1}  {op}  {num2} = ?",
                "num1": num1,
                "operator": op,
                "num2": num2,
                "answer": ans
            })
        return questions

    def _build_ui(self):
        """Construct the graphical user interface components."""
        # Top Header Frame
        header_frame = tk.Frame(self.root, bg="#1e293b", height=70)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="Mental Fatigue Assessment Study",
            font=("Helvetica", 16, "bold"),
            fg="#f8fafc",
            bg="#1e293b"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

        self.progress_label = tk.Label(
            header_frame,
            text=f"Question 0 / {self.total_questions}",
            font=("Helvetica", 12),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.progress_label.pack(side=tk.RIGHT, padx=20, pady=15)

        # Main Workspace Card
        self.card_frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.GROOVE)
        self.card_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=25)

        self.prompt_label = tk.Label(
            self.card_frame,
            text="Please calculate the answer below:",
            font=("Helvetica", 12),
            fg="#475569",
            bg="#ffffff"
        )
        self.prompt_label.pack(pady=(20, 10))

        self.question_display = tk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 28, "bold"),
            fg="#0f172a",
            bg="#ffffff"
        )
        self.question_display.pack(pady=15)

        # Answer Entry Box
        entry_frame = tk.Frame(self.card_frame, bg="#ffffff")
        entry_frame.pack(pady=15)

        entry_sublabel = tk.Label(
            entry_frame,
            text="Your Answer: ",
            font=("Helvetica", 14),
            bg="#ffffff"
        )
        entry_sublabel.pack(side=tk.LEFT, padx=5)

        self.answer_entry = tk.Entry(
            entry_frame,
            font=("Helvetica", 16, "bold"),
            width=10,
            justify="center",
            bd=2,
            relief=tk.SOLID
        )
        self.answer_entry.pack(side=tk.LEFT, padx=5)
        self.answer_entry.focus_set()

        # Submit Button
        self.submit_btn = tk.Button(
            self.card_frame,
            text="Submit Answer [Enter]",
            font=("Helvetica", 12, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            padx=15,
            pady=8,
            command=self._handle_submit
        )
        self.submit_btn.pack(pady=15)

        # Break & Control Bar
        control_frame = tk.Frame(self.card_frame, bg="#ffffff")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15, padx=20)

        self.break_btn = tk.Button(
            control_frame,
            text="☕ Take a short break",
            font=("Helvetica", 11),
            bg="#f59e0b",
            fg="#ffffff",
            activebackground="#d97706",
            activeforeground="#ffffff",
            command=self._toggle_break
        )
        self.break_btn.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            control_frame,
            text="Status: Active",
            font=("Helvetica", 11, "italic"),
            fg="#059669",
            bg="#ffffff"
        )
        self.status_label.pack(side=tk.RIGHT)

    def _bind_events(self):
        """Bind mouse movement, click, keystroke, and enter-key events."""
        # Keystroke binding on Entry widget
        self.answer_entry.bind("<KeyPress>", lambda e: self._log_keystroke(e, "press"))
        self.answer_entry.bind("<KeyRelease>", lambda e: self._log_keystroke(e, "release"))
        self.answer_entry.bind("<Return>", lambda e: self._handle_submit())

        # Global Mouse motion (sampled at 50ms interval) and click tracking
        self.root.bind("<Motion>", self._log_mouse_motion)
        self.root.bind("<Button-1>", self._log_mouse_click)

    def _log_keystroke(self, event: tk.Event, event_type: str):
        """Log key press/release events with high-precision timestamp."""
        if self.is_on_break:
            return

        curr_time = time.time()
        key_symbol = event.keysym
        char = event.char

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
        """Log mouse movement sampled every ~50ms to minimize log overhead."""
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
        """Log mouse click positions and target widget elements."""
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
        """Handle starting and resuming voluntary user breaks."""
        curr_time = time.time()

        if not self.is_on_break:
            # Start Break
            self.is_on_break = True
            self.break_start_time = curr_time
            self.break_btn.config(text="▶ Resume Quiz", bg="#10b981", activebackground="#059669")
            self.status_label.config(text="Status: On Break ⏸", fg="#d97706")
            self.answer_entry.config(state=tk.DISABLED)
            self.submit_btn.config(state=tk.DISABLED)
        else:
            # End Break
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
            self.break_btn.config(text="☕ Take a short break", bg="#f59e0b", activebackground="#d97706")
            self.status_label.config(text="Status: Active", fg="#059669")
            self.answer_entry.config(state=tk.NORMAL)
            self.submit_btn.config(state=tk.NORMAL)
            self.answer_entry.focus_set()

    def _handle_submit(self):
        """Process user answer submission for current question."""
        if self.is_on_break:
            return

        user_raw = self.answer_entry.get().strip()
        if not user_raw:
            messagebox.showwarning("Input Required", "Please enter an answer before submitting.")
            return

        curr_time = time.time()
        response_time = curr_time - self.question_start_time
        elapsed_session_time = curr_time - self.session_start_time

        q_info = self.questions[self.current_q_idx]
        correct_ans = q_info["answer"]

        try:
            user_ans = int(user_raw)
            is_correct = 1 if (user_ans == correct_ans) else 0
        except ValueError:
            user_ans = user_raw
            is_correct = 0

        # Log performance
        self.performance_logs.append({
            "session_id": self.session_id,
            "question_index": q_info["q_num"],
            "question_text": q_info["text"],
            "correct_answer": correct_ans,
            "user_answer": user_ans,
            "is_correct": is_correct,
            "response_time_sec": response_time,
            "elapsed_session_time_sec": elapsed_session_time,
            "difficulty_tier": self.difficulty_tier
        })

        self.current_q_idx += 1

        # Check if fatigue self-report prompt should pop up (every 10 questions)
        if self.current_q_idx % 10 == 0:
            self._show_fatigue_prompt()

        # Check if quiz complete
        if self.current_q_idx >= self.total_questions:
            self._finish_session()
        else:
            self._load_next_question()

    def _show_fatigue_prompt(self):
        """Show Karolinska Sleepiness Scale modal fatigue prompt (1–7)."""
        modal = tk.Toplevel(self.root)
        modal.title("Fatigue Assessment Checkpoint")
        modal.geometry("450x380")
        modal.grab_set()  # Make window modal
        modal.resizable(False, False)

        tk.Label(
            modal,
            text=f"Checkpoint: Question {self.current_q_idx} / {self.total_questions}",
            font=("Helvetica", 14, "bold"),
            fg="#1e293b"
        ).pack(pady=(20, 10))

        tk.Label(
            modal,
            text="Rate your current mental fatigue:",
            font=("Helvetica", 12),
            fg="#475569"
        ).pack(pady=5)

        selected_rating = tk.IntVar(value=4)

        ratings_desc = [
            (1, "1 — Extremely Alert / Fully Awake"),
            (2, "2 — Very Alert"),
            (3, "3 — Alert / Normal State"),
            (4, "4 — Fairly Alert"),
            (5, "5 — Neither Alert nor Sleepy"),
            (6, "6 — Some Signs of Fatigue"),
            (7, "7 — Extremely Fatigued / Struggling to focus")
        ]

        frame_radios = tk.Frame(modal)
        frame_radios.pack(pady=10, px=20, anchor=tk.W)

        for val, desc in ratings_desc:
            tk.Radiobutton(
                frame_radios,
                text=desc,
                variable=selected_rating,
                value=val,
                font=("Helvetica", 10),
                anchor=tk.W
            ).pack(fill=tk.X, pady=2)

        def save_and_close():
            score = selected_rating.get()
            curr_time = time.time()
            self.fatigue_rating_logs.append({
                "session_id": self.session_id,
                "question_checkpoint": self.current_q_idx,
                "timestamp": curr_time,
                "datetime_iso": datetime.datetime.fromtimestamp(curr_time).isoformat(),
                "fatigue_score": score
            })
            modal.destroy()

        submit_rating_btn = tk.Button(
            modal,
            text="Confirm Rating",
            font=("Helvetica", 11, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            command=save_and_close
        )
        submit_rating_btn.pack(pady=15)

        self.root.wait_window(modal)

    def _load_next_question(self):
        """Prepare UI for the next question."""
        self.progress_label.config(text=f"Question {self.current_q_idx + 1} / {self.total_questions}")
        q_info = self.questions[self.current_q_idx]
        self.question_display.config(text=q_info["text"])
        self.answer_entry.delete(0, tk.END)
        self.question_start_time = time.time()

    def _finish_session(self):
        """Save all 5 CSV logs and close application."""
        self._export_csvs()
        messagebox.showinfo(
            "Session Completed",
            f"Session {self.session_id} completed successfully!\nAll 5 CSV log files saved to:\n{self.output_dir}"
        )
        self.root.destroy()

    def _export_csvs(self):
        """Write all 5 collected data logs to CSV files with schema guarantees."""
        schemas = {
            f"keystrokes_{self.session_id}.csv": (self.keystroke_logs, ["session_id", "question_index", "timestamp", "datetime_iso", "event_type", "key_symbol", "is_backspace"]),
            f"mouse_events_{self.session_id}.csv": (self.mouse_event_logs, ["session_id", "question_index", "timestamp", "event_type", "x", "y", "target_element"]),
            f"performance_{self.session_id}.csv": (self.performance_logs, ["session_id", "question_index", "question_text", "correct_answer", "user_answer", "is_correct", "response_time_sec", "elapsed_session_time_sec", "difficulty_tier"]),
            f"fatigue_ratings_{self.session_id}.csv": (self.fatigue_rating_logs, ["session_id", "question_checkpoint", "timestamp", "datetime_iso", "fatigue_score"]),
            f"breaks_{self.session_id}.csv": (self.break_logs, ["session_id", "question_index", "break_start_time", "break_end_time", "duration_sec"]),
        }

        for filename, (data, cols) in schemas.items():
            filepath = os.path.join(self.output_dir, filename)
            df = pd.DataFrame(data) if data else pd.DataFrame(columns=cols)
            df.to_csv(filepath, index=False)
            print(f"[EXPORT] Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FatigueDataCollectorApp(root, total_questions=50)
    root.mainloop()
