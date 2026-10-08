# Mental Fatigue Detection — Multi-Signal System & Desktop Station

An end-to-end Python system for detecting real-time mental fatigue during timed cognitive tasks using a validated multi-signal architecture (keystroke dynamics, mouse movement, task performance, error taxonomy, break behavior, and time-on-task) validated against self-reported Karolinska Sleepiness Scale ground truth.

Includes a **Desktop Tkinter Application** for typing tasks, live behavioral tracking, in-app ML analytics, and automated clean dataset exports for **Microsoft Power BI** dashboards.

---

## 📌 Architecture & System Flow

```
                  +----------------------------------------------+
                  |         Tkinter Desktop Station              |
                  |  (Live Arithmetic & Multi-Signal Logging)    |
                  +----------------------------------------------+
                                          |
     +-------------------+---------------+-------------------+-------------------+
     |                   |               |                   |                   |
Keystrokes           Mouse Events   Performance Logs   Fatigue Prompts      Break Events
(press/release)      (motion/click)  (accuracy/RT)      (Karolinska 1-7)    (start/duration)
     |                   |               |                   |                   |
     +-------------------+---------------+-------------------+-------------------+
                                          |
                                          v
                         +-------------------------------+
                         |   Window Feature Engineering  |
                         |  (10-question checkpoints)    |
                         +-------------------------------+
                                          |
                                          v
                         +-------------------------------+
                         |      XGBoost Model Pipeline   |
                         |  (5-Fold Stratified CV, SHAP) |
                         +-------------------------------+
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
    +-------------------------------+           +-------------------------------+
    |   Tkinter Desktop Analytics   |           |    Power BI Desktop Reports   |
    | (Real-time in-app dashboard)  |           |  (Star Schema & DAX Measures) |
    +-------------------------------+           +-------------------------------+
```

---

## 📊 Signal Set (8 Core Signals)

| # | Signal Name | Category | Role | Description & Notes |
|---|-------------|----------|------|---------------------|
| 1 | **Karolinska Fatigue Score** | Ground Truth | **Label** | Self-reported 1–7 scale prompt every 10 questions |
| 2 | **Keystroke Dynamics** | Predictor | Motor Signal | Mean press duration, inter-key pause, backspace rate, typing speed, rhythm variability |
| 3 | **Task Performance** | Predictor | Cognitive Signal | Accuracy rate, mean response time, response time variance |
| 4 | **Time-on-Task** | Predictor | Temporal Signal | Cumulative elapsed session duration at each checkpoint |
| 5 | **Error Pattern Taxonomy** | Predictor | Cognitive Strategy | Careless errors (fast & wrong) vs Effortful errors (slow & wrong) |
| 6 | **Mouse Movement & Clicks** | Predictor | Motor Signal | Movement speed, direction change jitter, click precision, click rate |
| 7 | **Voluntary Break Behavior** | Predictor | Self-Regulatory | Frequency and cumulative duration of self-initiated pauses |
| 8 | **Question Difficulty** | Controlled | Control Variable | Logged explicitly (`Tier 2 - Medium`) to rule out task difficulty confounds |

---

## 🛠️ Project Structure

```
mental fatigue detection/
│
├── data/                                 # Raw session CSVs & SHAP plots
│   ├── keystrokes_<session_id>.csv
│   ├── mouse_events_<session_id>.csv
│   ├── performance_<session_id>.csv
│   ├── fatigue_ratings_<session_id>.csv
│   ├── breaks_<session_id>.csv
│   └── power_bi_export/                 # Normalized Power BI tables
│       ├── power_bi_sessions_summary.csv
│       ├── power_bi_window_features.csv
│       ├── power_bi_question_performance.csv
│       └── power_bi_feature_importance.csv
│
├── src/
│   ├── __init__.py
│   ├── tkinter_dashboard.py              # Unified Desktop Tkinter GUI Application
│   ├── data_collection.py                # Standalone Tkinter task
│   ├── feature_engineering.py            # Stage 2: 10-question window feature extraction
│   ├── label_preparation.py              # Stage 3: Continuous vs binary target formatting
│   ├── model_training.py                 # Stage 4: XGBoost, 5-Fold Stratified CV, ANOVA & SHAP
│   ├── power_bi_export.py                # Power BI dataset generator
│   └── generate_sample_data.py           # Utility: Automated multi-session data generator
│
├── app.py                                # Desktop application launcher
├── main.py                               # Standard root launcher
├── power_bi_dashboard_guide.md           # Power BI build guide, Star Schema & DAX formulas
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## ⚡ Quickstart Guide

### 1. Installation & Environment Setup
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Desktop Application (Typing + Analytics + Power BI Exporter)
```bash
python app.py
```
*or*
```bash
python main.py
```

Inside the application:
- **Tab 1 (Typing & Cognitive Assessment)**: Click **Start New Session (50 Qs)** or **Quick Test (10 Qs)** to start the interactive arithmetic task with live keystroke/mouse tracking.
- **Tab 2 (Fatigue Analytics & ML Model)**: Inspect session predictions, multi-signal degradation curves, and SHAP top drivers.
- **Tab 3 (Power BI Export Center)**: Click **Export All Datasets for Power BI** to create clean CSVs for Power BI.

---

### 3. Run Pipeline via Terminal (Optional)

```bash
# Step A: (Optional) Generate simulated benchmark sessions
python src/generate_sample_data.py

# Step B: Extract 18 multi-signal features across 10-question windows
python src/feature_engineering.py

# Step C: Prepare labels
python src/label_preparation.py

# Step D: Train XGBoost, run Stratified CV, ANOVA & SHAP
python src/model_training.py

# Step E: Export Power BI datasets
python src/power_bi_export.py
```

---

## 📈 Power BI Dashboard Integration

All exported tables are stored in `data/power_bi_export/`.

Refer to [`power_bi_dashboard_guide.md`](power_bi_dashboard_guide.md) for:
- Star schema relationship diagrams (1-to-many connections).
- DAX measures (`Avg Actual Fatigue`, `Avg Predicted Fatigue`, `Prediction Gap`, `Careless Error Rate %`, `Fatigue Status Alert`).
- Recommended 4-page dashboard visuals and color palettes.

---

## 🔬 Empirical Results & Model Performance

- **Primary Modeling Strategy**: Continuous XGBoost Regression ($y \in [1, 7]$)
- **Out-of-Fold $R^2$ (5-Fold Stratified CV)**: `0.9380`
- **Out-of-Fold RMSE**: `0.3521`
- **Out-of-Fold MAE**: `0.1376`
- **Top ANOVA Feature Drivers**: `mean_mouse_speed` ($p < 10^{-30}$), `mean_press_dur` ($p < 10^{-26}$), `time_on_task` ($p < 10^{-25}$).
