import pandas as pd

df = pd.read_csv("students.csv")

for col in df.columns:
    if df[col].astype(str).str.contains(r'\?').any():
        print(f"Column '{col}' contains '?' values")
