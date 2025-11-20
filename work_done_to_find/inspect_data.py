import pandas as pd

df = pd.read_csv("students.csv")

print("\n--- Shape of dataset ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns)

print("\n--- Missing values ---")
print(df.isna().sum())

print("\n--- Data sample ---")
print(df.head(10))
