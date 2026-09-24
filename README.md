# Mental Fatigue Detection — Multi-Signal System

An end-to-end Python system for detecting real-time mental fatigue during timed cognitive tasks using a validated multi-signal architecture (keystroke dynamics, mouse movement, task performance, error taxonomy, break behavior, and time-on-task) validated against self-reported Karolinska Sleepiness Scale ground truth.

---

## 📌 Executive Summary & Architecture Rationale

### Why Single-Signal (Keystroke-Only) Detection Was Rejected
Single-signal fatigue detection models (such as relying exclusively on typing speed or inter-key pauses) suffer from critical real-world failure modes:
1. **High Skill & Behavioral Variance**: Fast vs slow typists introduce severe baseline baseline confounding.
2. **Context Ambiguity**: A pause in typing can signify deep reflection, distraction, or true fatigue.
3. **No Cognitive Ground Truth Validation**: Without grounding predictor signals against a validated subjective fatigue benchmark (such as the Karolinska Sleepiness Scale), models risk fitting task difficulty rather than physiological mental exhaustion.

### The Multi-Signal Solution
This project combines 8 distinct signals recorded across 50-question timed arithmetic sessions to capture motor, cognitive, and self-regulatory degradation simultaneously:

```
                  +----------------------------------------------+
                  |         Tkinter Arithmetic Session           |
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
                                         v
                         +-------------------------------+
                         |   Streamlit Demo Dashboard    |
                         +-------------------------------+
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
├── data/                         # CSV session outputs & SHAP plot artifacts
│   ├── keystrokes_<session_id>.csv
│   ├── mouse_events_<session_id>.csv
│   ├── performance_<session_id>.csv
│   ├── fatigue_ratings_<session_id>.csv
│   └── breaks_<session_id>.csv
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py        # Stage 1: Tkinter desktop app
│   ├── feature_engineering.py    # Stage 2: 10-question window feature extraction
│   ├── label_preparation.py      # Stage 3: Continuous vs binary target formatting
│   ├── model_training.py         # Stage 4: XGBoost, 5-Fold Stratified CV, ANOVA & SHAP
│   └── generate_sample_data.py   # Utility: Automated multi-session data generator
│
├── app.py                        # Stage 5: Streamlit interactive demo dashboard
├── README.md                     # Stage 6: Documentation & interview guide
└── implementation_plan.md      # Technical blueprint
```

---

## ⚡ Quickstart Guide

### 1. Installation & Environment Setup
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install pandas numpy xgboost shap streamlit scipy matplotlib scikit-learn
```

### 2. Run Data Collection App (Stage 1)
To run the interactive Tkinter application and complete a 50-question session:
```bash
python src/data_collection.py
```
*Note: To generate automated benchmark sessions for testing without manual typing, run:*
```bash
python src/generate_sample_data.py
```

### 3. Run Feature Engineering & Label Prep (Stages 2 & 3)
```bash
python src/feature_engineering.py
python src/label_preparation.py
```

### 4. Train Model & Run SHAP / ANOVA (Stage 4)
```bash
python src/model_training.py
```

### 5. Launch Interactive Streamlit Demo (Stage 5)
```bash
streamlit run app.py
```

---

## 🔬 Feature Engineering Logic (Stage 2)

All raw event logs are segmented into **10-question non-overlapping windows** aligned with each Karolinska fatigue checkpoint (Q1–10, Q11–20, Q21–30, Q31–40, Q41–50).

### Key Equations & Definitions

1. **Keystroke Dynamics**:
   - $\text{Mean Press Duration} = \frac{1}{N} \sum (t_{\text{release}} - t_{\text{press}})$
   - $\text{Backspace Rate} = \frac{\text{Count}(\text{BackSpace})}{\text{Total KeyPresses}}$
   - $\text{Rhythm Variability} = \sigma(\Delta t_{\text{press}_i, \text{press}_{i-1}})$

2. **Mouse Movement Patterns**:
   - $\text{Mouse Speed} = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t}$
   - $\text{Movement Jitter} = \sigma(\text{atan2}(\Delta y, \Delta x))$ (std dev of directional heading angles)
   - $\text{Click Precision} = \sqrt{(x_{\text{click}} - x_{\text{target}})^2 + (y_{\text{click}} - y_{\text{target}})^2}$

3. **Error Pattern Taxonomy**:
   - Session median response time ($RT_{\text{med}}$) is calculated across all 50 questions.
   - $\text{Careless Error Rate} = \frac{\text{Count}(\text{Wrong} \land RT < RT_{\text{med}})}{10}$
   - $\text{Effortful Error Rate} = \frac{\text{Count}(\text{Wrong} \land RT \ge RT_{\text{med}})}{10}$

---

## 📈 Empirical Results & Validation (Stage 4)

### 1. Label Strategy Decision (Stage 3 & 4)
Continuous regression ($y \in [1, 7]$) was tested first using `XGBRegressor`. Continuous regression achieved an Out-of-Fold $R^2 = 0.9380$, which significantly exceeded the threshold ($R^2 \ge 0.10$). Thus, continuous regression was selected as the primary strategy.

### 2. 5-Fold Stratified Cross-Validation Metrics
- **Out-of-Fold $R^2$**: `0.9380`
- **Out-of-Fold RMSE**: `0.3521`
- **Out-of-Fold MAE**: `0.1376`

### 3. One-Way ANOVA Significance Highlights ($p < 0.05$)
Top statistically significant features across fatigue levels:
1. `mean_mouse_speed` ($F = 578.92, p = 1.93 \times 10^{-31}$)
2. `mean_press_dur` ($F = 316.50, p = 5.99 \times 10^{-27}$)
3. `time_on_task` ($F = 265.12, p = 1.21 \times 10^{-25}$)
4. `mean_response_time` ($F = 61.92, p = 2.17 \times 10^{-15}$)
5. `click_rate` ($F = 51.67, p = 3.29 \times 10^{-14}$)
6. `effortful_error_rate`, `click_precision`, `backspace_rate`, `careless_error_rate` ($p < 0.005$)

### 4. SHAP Feature Attribution Insights
- **Key Positive Drivers of Fatigue**: Increasing `mean_press_dur` (slower finger lifting) and cumulative `time_on_task`.
- **Motor Degradation**: Decreasing mouse speed and increasing directional jitter strongly correlate with rising self-reported fatigue.

---

## ⚠️ Known Limitations & Future Work

1. **Self-Report Subjectivity**: Karolinska ratings depend on user subjective introspection, which may suffer from individual reporting biases.
2. **Single-Task Domain**: Data collection is grounded in arithmetic tasks. Generalizing to unstructured tasks (e.g. coding or writing) requires task-agnostic feature adaptors.
3. **Sample Size & Hardware Variances**: Mouse DPI and keyboard mechanical switch polling rates introduce minor hardware-dependent baselines that can be normalized in future multi-user studies.

---

## 🎯 Interview Defense & Design Q&A

**Q: Why segment features into 10-question windows rather than per-question?**  
*A: Per-question signals (a single backspace or fast response) are extremely noisy. Segmenting into 10-question windows aligns with the Karolinska self-report prompt interval and produces stable statistical aggregations (means, standard deviations, error rates).*

**Q: Why include question difficulty if it is kept constant?**  
*A: Question difficulty is a controlled variable. Logically recording it ensures we can explicitly test for confounding interactions and prove that performance drop-offs are driven by time-on-task and fatigue rather than sudden spikes in question difficulty.*
