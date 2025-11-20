import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("students.csv")
df.replace("?", np.nan, inplace=True)

# Fix numeric columns
numeric_cols = ["Attendance", "StudyHours", "Marks", "PreviousScore", "FinalScore"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill blanks
df["Attendance"] = df["Attendance"].fillna(0)
df["StudyHours"] = df["StudyHours"].fillna(df["StudyHours"].mean())
df["Marks"] = df["Marks"].fillna(0)
df["PreviousScore"] = df["PreviousScore"].fillna(df["PreviousScore"].mean())

# Clean ParentEdu
mapping = {"Bachelor":"UniDegree","BSc":"UniDegree","University":"UniDegree","Uni":"UniDegree",
           "HighSchool":"HighSchool","School":"School","NoEdu":"NoEdu"}
df["ParentEdu"] = df["ParentEdu"].replace(mapping)
df["ParentEdu"] = df["ParentEdu"].map({"NoEdu":0,"School":1,"HighSchool":2,"UniDegree":3})

# Encode TestPrep
df["TestPrep"] = df["TestPrep"].map({"yes":1, "no":0})

# Features for regression
features = ["Attendance", "StudyHours", "Marks", "PreviousScore", "TestPrep"]
X = df[features]
y = df["FinalScore"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Print feature importance
for feat, coef in zip(features, model.coef_):
    print(f"{feat}: {coef}")
