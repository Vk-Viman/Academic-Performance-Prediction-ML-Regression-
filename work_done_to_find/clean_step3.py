import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Standardize ParentEdu column
df["ParentEdu"] = df["ParentEdu"].replace({
    "Bachelor": "UniDegree",
    "BSc": "UniDegree",
    "University": "UniDegree",
    "Uni": "UniDegree",
    "HighSchool": "HighSchool",
    "School": "School",
    "NoEdu": "NoEdu"
})

print(df["ParentEdu"].unique())
