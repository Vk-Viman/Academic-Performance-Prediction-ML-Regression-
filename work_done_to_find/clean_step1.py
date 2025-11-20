import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Replace "?" with actual missing value NaN
df.replace("?", np.nan, inplace=True)

print(df.head(10))
print("\nMissing values after cleaning:")
print(df.isna().sum())
