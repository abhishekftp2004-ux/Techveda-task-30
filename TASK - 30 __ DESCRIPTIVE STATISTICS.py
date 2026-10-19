"""
Task 30 — Descriptive Statistics
Dataset: student_performance.csv
Requirements: Python, pandas, NumPy
Run: python descriptive_statistics.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_FILE = Path(__file__).with_name("student_performance.csv")
df = pd.read_csv(DATA_FILE)

# Select at least three numerical variables for analysis.
numeric_cols = ["Study_Hours", "Attendance_Percent", "Exam_Score"]
numeric = df[numeric_cols]

print("=" * 72)
print("TASK 30: DESCRIPTIVE STATISTICS — STUDENT PERFORMANCE")
print("=" * 72)
print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst five records:")
print(df.head().to_string(index=False))

# describe() gives count, mean, standard deviation, min, quartiles, and max.
print("\n1) QUICK SUMMARY (Pandas describe):")
print(numeric.describe().round(2).to_string())

# Explicit calculation of central tendency and dispersion.
summary = pd.DataFrame({
    "Mean": numeric.mean(),
    "Median": numeric.median(),
    "Mode": numeric.mode().iloc[0],
    "Range": numeric.max() - numeric.min(),
    "Variance_Sample": numeric.var(ddof=1),
    "Std_Dev_Sample": numeric.std(ddof=1),
    "Q1_25%": numeric.quantile(0.25),
    "Q2_50%": numeric.quantile(0.50),
    "Q3_75%": numeric.quantile(0.75),
    "IQR": numeric.quantile(0.75) - numeric.quantile(0.25),
    "P10": numeric.quantile(0.10),
    "P90": numeric.quantile(0.90),
}).round(2)

print("\n2) CENTRAL TENDENCY, DISPERSION, QUARTILES & PERCENTILES:")
print(summary.to_string())

print("\n3) MISSING VALUES:")
print(df.isna().sum().to_string())

# Compare mean and median to help assess skew/outlier influence.
print("\n4) MEAN VS MEDIAN INTERPRETATION:")
for col in numeric_cols:
    mean_val, median_val = numeric[col].mean(), numeric[col].median()
    if np.isclose(mean_val, median_val, atol=0.5):
        note = "mean and median are close; distribution may be roughly balanced."
    elif mean_val > median_val:
        note = "mean exceeds median; higher values may be pulling the mean upward."
    else:
        note = "median exceeds mean; lower values may be pulling the mean downward."
    print(f"- {col}: mean={mean_val:.2f}, median={median_val:.2f}; {note}")

# Export a tidy summary table for submission.
summary.to_csv(Path(__file__).with_name("descriptive_summary.csv"), index_label="Variable")
print("\nSaved: descriptive_summary.csv")
