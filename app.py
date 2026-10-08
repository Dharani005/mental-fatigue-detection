"""
Mental Fatigue Detection — Desktop Application Launcher.

Launches the unified Tkinter Desktop Application containing:
1. Live Arithmetic Typing & Cognitive Assessment Station (Keystrokes, Mouse, Karolinska prompts)
2. Desktop Fatigue Analytics & ML Prediction Explorer (XGBoost, SHAP, ANOVA)
3. Power BI Data Export Center & Modeling Schema Guide
"""

import sys
import tkinter as tk
from src.tkinter_dashboard import MentalFatigueApp


def main():
    root = tk.Tk()
    app = MentalFatigueApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
