# Power BI Dashboard Architecture & Build Guide — Mental Fatigue Detection

This guide provides step-by-step instructions for importing, modeling, and creating an executive-grade Power BI dashboard using the exported multi-signal mental fatigue dataset.

---

## 📊 1. Data Architecture & Star Schema

The system exports 4 clean, pre-processed CSV tables into `data/power_bi_export/`.

```
                  +-----------------------------------+
                  |   power_bi_sessions_summary       |  (Dimension Table)
                  |   PK: session_id                  |
                  +-----------------------------------+
                                    | 1
                                    |
                    +---------------+---------------+
                  * |                               | *
+-----------------------------------+   +------------------------------------+
|   power_bi_window_features        |   |  power_bi_question_performance     |  (Fact Tables)
|   FK: session_id                  |   |  FK: session_id                    |
|   Key: session_id + checkpoint    |   |  Key: session_id + question_index  |
+-----------------------------------+   +------------------------------------+

                  +-----------------------------------+
                  |   power_bi_feature_importance     |  (Summary Dimension)
                  |   PK: feature                     |
                  +-----------------------------------+
```

### Table Inventory & Schema Reference

| Table Name | Role | Primary/Join Key | Key Columns & Metrics |
|------------|------|------------------|-----------------------|
| `power_bi_sessions_summary.csv` | **Dimension** | `session_id` | `session_id`, `total_questions`, `overall_accuracy_pct`, `mean_response_time_sec`, `total_duration_sec`, `initial_fatigue_score`, `final_fatigue_score`, `fatigue_score_change`, `total_breaks_taken`, `total_break_duration_sec`, `final_fatigue_category` |
| `power_bi_window_features.csv` | **Fact** | `session_id` + `window_checkpoint` | `window_checkpoint`, `fatigue_score` (Actual 1-7), `predicted_fatigue_score`, `prediction_error`, `mean_press_dur`, `backspace_rate`, `typing_speed_kps`, `mean_mouse_speed`, `movement_jitter`, `click_precision`, `accuracy`, `mean_response_time`, `careless_error_rate`, `effortful_error_rate`, `actual_fatigue_category` |
| `power_bi_question_performance.csv` | **Fact** | `session_id` + `question_index` | `question_index`, `question_text`, `correct_answer`, `user_answer`, `is_correct`, `response_time_sec`, `elapsed_session_time_sec`, `is_careless_error`, `is_effortful_error` |
| `power_bi_feature_importance.csv` | **Summary** | `feature` | `feature`, `f_statistic`, `p_value`, `significant_p05`, `mean_abs_shap_impact`, `importance_rank` |

---

## ⚡ 2. How to Export the Datasets

You can export the latest clean Power BI datasets in two ways:

1. **Via the Tkinter Desktop Application**:
   - Run `python app.py` (or `python main.py`).
   - Go to the **📈 Power BI Export Center** tab.
   - Click **🚀 Export All Datasets for Power BI**.

2. **Via Terminal**:
   ```bash
   python src/power_bi_export.py
   ```

Files will be placed in `data/power_bi_export/`.

---

## 🛠️ 3. Power BI Desktop Step-by-Step Setup

### Step A: Load Data
1. Open **Microsoft Power BI Desktop**.
2. Click **Get Data** > **Text/CSV**.
3. Select and load all 4 files from `data/power_bi_export/`:
   - `power_bi_sessions_summary.csv`
   - `power_bi_window_features.csv`
   - `power_bi_question_performance.csv`
   - `power_bi_feature_importance.csv`
4. Click **Load**.

### Step B: Establish Relationships (Model View)
Go to the **Model View** tab in Power BI:
1. Drag `session_id` from `power_bi_sessions_summary` to `session_id` in `power_bi_window_features` (Relationship: **1 to Many (1:*)**, Single Cross filter direction).
2. Drag `session_id` from `power_bi_sessions_summary` to `session_id` in `power_bi_question_performance` (Relationship: **1 to Many (1:*)**, Single Cross filter direction).

---

## 📐 4. Essential DAX Measures

Create a new dedicated table in Power BI named `_Measures` and add the following DAX calculations:

### 1. Actual Mean Fatigue Score
```dax
Avg Actual Fatigue = AVERAGE(power_bi_window_features[fatigue_score])
```

### 2. Predicted Mean Fatigue Score
```dax
Avg Predicted Fatigue = AVERAGE(power_bi_window_features[predicted_fatigue_score])
```

### 3. Model Prediction Gap (MAE)
```dax
Fatigue Prediction Gap = 
AVERAGEX(
    power_bi_window_features, 
    ABS(power_bi_window_features[predicted_fatigue_score] - power_bi_window_features[fatigue_score])
)
```

### 4. Overall Accuracy %
```dax
Average Accuracy % = AVERAGE(power_bi_sessions_summary[overall_accuracy_pct])
```

### 5. Careless vs Effortful Error Distribution
```dax
Careless Error Rate % = 
DIVIDE(
    SUM(power_bi_question_performance[is_careless_error]),
    COUNTROWS(power_bi_question_performance),
    0
) * 100
```

```dax
Effortful Error Rate % = 
DIVIDE(
    SUM(power_bi_question_performance[is_effortful_error]),
    COUNTROWS(power_bi_question_performance),
    0
) * 100
```

### 6. Dynamic Fatigue Alert Status
```dax
Fatigue Status Alert = 
VAR CurrentScore = [Avg Predicted Fatigue]
RETURN
    SWITCH(
        TRUE(),
        CurrentScore >= 5.5, "🔴 Severe Fatigue - Rest Urgently Recommended",
        CurrentScore >= 4.0, "🟡 Moderate Fatigue - Break Advised",
        "🟢 Alert & Optimal Performance"
    )
```

---

## 🎨 5. Recommended Power BI Dashboard Layout (4 Pages)

### Page 1: 📊 Executive Fatigue & Performance Overview
- **Top Slicer**: `session_id` dropdown / tile list.
- **KPI Cards (Top Ribbon)**:
  - Card 1: `Avg Actual Fatigue` (Scale 1–7)
  - Card 2: `Avg Predicted Fatigue` (Scale 1–7)
  - Card 3: `Average Accuracy %`
  - Card 4: `Average Response Time`
  - Card 5: `Fatigue Status Alert` (KPI / Status indicator)
- **Visual 1 (Line Chart)**: `window_checkpoint` on X-axis vs `Avg Actual Fatigue` and `Avg Predicted Fatigue` on Y-axis.
- **Visual 2 (Clustered Column Chart)**: Fatigue Score change by Session ID.
- **Visual 3 (Donut Chart)**: `final_fatigue_category` distribution.

### Page 2: ⌨️ Motor & Keystroke Dynamics Degradation
- **Visual 1 (Line Chart)**: `window_checkpoint` on X-axis vs `mean_press_dur` (Key hold duration) and `typing_speed_kps`.
- **Visual 2 (Scatter Plot)**: X-axis: `mean_press_dur`, Y-axis: `mean_mouse_speed`, Legend: `actual_fatigue_category`.
- **Visual 3 (Line Chart)**: Backspace correction rate (`backspace_rate`) across question checkpoints.

### Page 3: 🧠 Cognitive Error Patterns & Reaction Latency
- **Visual 1 (Stacked Bar Chart)**: Total Errors split by `is_careless_error` (Fast & Wrong) vs `is_effortful_error` (Slow & Wrong) over checkpoints.
- **Visual 2 (Histogram / Area Chart)**: `response_time_sec` distribution comparing Alert vs Fatigued windows.
- **Visual 3 (Gauge Chart)**: Voluntary break frequency and rest duration.

### Page 4: 🔍 Machine Learning & SHAP Attribution
- **Visual 1 (Horizontal Bar Chart)**: `feature` vs `mean_abs_shap_impact` sorted descending (Top Fatigue Drivers).
- **Visual 2 (Table Visual)**: `feature`, `f_statistic`, `p_value`, `significant_p05`, `importance_rank`.
- **Visual 3 (Scatter Plot)**: Actual Karolinska vs Predicted Score with a 45-degree reference line ($R^2 = 0.938$).

---

## 🎯 6. Color Palette Recommendations

- **Alert / Low Fatigue**: `#10b981` (Emerald Green)
- **Moderate Alertness**: `#3b82f6` (Royal Blue)
- **Mild Fatigue**: `#f59e0b` (Amber Orange)
- **Severe Fatigue**: `#ef4444` (Crimson Red)
- **Background**: `#f8fafc` (Slate Light) / Dark Mode `#0f172a`
