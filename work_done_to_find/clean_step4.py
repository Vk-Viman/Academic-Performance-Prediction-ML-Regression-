import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Standardize ParentEdu text
df["ParentEdu"] = df["ParentEdu"].replace({
    "Bachelor": "UniDegree",
    "BSc": "UniDegree",
    "University": "UniDegree",
    "Uni": "UniDegree",
    "HighSchool": "HighSchool",
    "School": "School",
    "NoEdu": "NoEdu"
})

# Encode ParentEdu numerically (your mapping)
parentedu_map = {
    "NoEdu": 0,
    "School": 1,
    "HighSchool": 2,
    "UniDegree": 3
}

df["ParentEdu"] = df["ParentEdu"].map(parentedu_map)

print(df["ParentEdu"].unique())
print(df.head(10))
