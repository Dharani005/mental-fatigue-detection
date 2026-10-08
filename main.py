"""
Main Launcher for Mental Fatigue Detection System.

Runs the Tkinter Desktop Application with:
- Cognitive arithmetic typing station with real-time multi-signal tracking
- In-app ML analytics, SHAP feature attributions, and ANOVA significance
- Power BI dataset generation and schema documentation
"""

import tkinter as tk
from src.tkinter_dashboard import MentalFatigueApp


def main():
    root = tk.Tk()
    app = MentalFatigueApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
