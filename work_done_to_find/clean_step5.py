import pandas as pd
import numpy as np

df = pd.read_csv("students.csv")

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Standardize ParentEdu
df["ParentEdu"] = df["ParentEdu"].replace({
    "Bachelor": "UniDegree",
    "BSc": "UniDegree",
    "University": "UniDegree",
    "Uni": "UniDegree",
    "HighSchool": "HighSchool",
    "School": "School",
    "NoEdu": "NoEdu"
})

# Encode ParentEdu
parentedu_map = {
    "NoEdu": 0,
    "School": 1,
    "HighSchool": 2,
    "UniDegree": 3
}
df["ParentEdu"] = df["ParentEdu"].map(parentedu_map)

# Encode TestPrep (yes → 1, no → 0)
testprep_map = {"yes": 1, "no": 0}
df["TestPrep"] = df["TestPrep"].map(testprep_map)

print(df["TestPrep"].unique())
print(df.head(10))
