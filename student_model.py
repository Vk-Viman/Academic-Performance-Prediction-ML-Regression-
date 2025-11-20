import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1) Load data
df = pd.read_csv("students.csv")

# 2) Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# 3) Clean numeric columns: Attendance, StudyHours, Marks
df["Attendance"] = pd.to_numeric(df["Attendance"], errors="coerce")
df["StudyHours"] = pd.to_numeric(df["StudyHours"], errors="coerce")
df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")

# Fill NaNs
df["Attendance"] = df["Attendance"].fillna(0)
df["StudyHours"] = df["StudyHours"].fillna(df["StudyHours"].mean())
df["Marks"] = df["Marks"].fillna(0)

# 4) Standardize ParentEdu text
df["ParentEdu"] = df["ParentEdu"].replace({
    "Bachelor": "UniDegree",
    "BSc": "UniDegree",
    "University": "UniDegree",
    "Uni": "UniDegree",
    "HighSchool": "HighSchool",
    "School": "School",
    "NoEdu": "NoEdu"
})

# Encode ParentEdu with your mapping
parentedu_map = {
    "NoEdu": 0,
    "School": 1,
    "HighSchool": 2,
    "UniDegree": 3
}
df["ParentEdu"] = df["ParentEdu"].map(parentedu_map)

# 5) Encode TestPrep (yes/no → 1/0)
testprep_map = {"yes": 1, "no": 0}
df["TestPrep"] = df["TestPrep"].map(testprep_map)

# 6) Choose features (X) and target (y) for Pass/Fail classification
feature_cols = ["Attendance", "StudyHours", "Marks", "ParentEdu",
                "Absences", "FamilyIncome", "PreviousScore", "TestPrep"]

X = df[feature_cols]
y = df["Pass"]

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8) Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9) Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy on Pass/Fail:", accuracy)

# 10) Try a custom student prediction
new_student = pd.DataFrame([{
    "Attendance": 75,
    "StudyHours": 3,
    "Marks": 68,
    "ParentEdu": 3,      # UniDegree
    "Absences": 2,
    "FamilyIncome": 35000,
    "PreviousScore": 60,
    "TestPrep": 1        # did prep
}])

print("Prediction for new student (1=Pass, 0=Fail):", model.predict(new_student))
