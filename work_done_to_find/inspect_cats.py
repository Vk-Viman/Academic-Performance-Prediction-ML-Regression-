import pandas as pd

df = pd.read_csv("students.csv")
df.replace("?", None, inplace=True)

print("\nUnique values in ParentEdu:")
print(df["ParentEdu"].unique())

print("\nUnique values in TestPrep:")
print(df["TestPrep"].unique())
