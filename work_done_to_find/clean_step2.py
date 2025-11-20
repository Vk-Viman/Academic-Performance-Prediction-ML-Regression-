import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Step 1: Replace "?" strings with NaN
df.replace("?", np.nan, inplace=True)

# 1. Attendance: fill with 0
df["Attendance"] = df["Attendance"].astype(float)
df["Attendance"].fillna(0, inplace=True)

# 2. StudyHours: fill with mean
df["StudyHours"] = df["StudyHours"].astype(float)
df["StudyHours"].fillna(df["StudyHours"].mean(), inplace=True)

# 3. Marks: fill with 0
df["Marks"] = df["Marks"].astype(float)
df["Marks"].fillna(0, inplace=True)

print("\nMissing values after cleaning:")
print(df.isna().sum())
