import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Load data
# adjusted to new data folder
df = pd.read_csv("data/students.csv")

# 2) Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# 3) Convert numeric columns
df["Attendance"] = pd.to_numeric(df["Attendance"], errors="coerce")
df["StudyHours"] = pd.to_numeric(df["StudyHours"], errors="coerce")
df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
df["PreviousScore"] = pd.to_numeric(df["PreviousScore"], errors="coerce")
df["FinalScore"] = pd.to_numeric(df["FinalScore"], errors="coerce")

# 4) Fill missing numeric values (your choices)
df["Attendance"] = df["Attendance"].fillna(0)
df["StudyHours"] = df["StudyHours"].fillna(df["StudyHours"].mean())
df["Marks"] = df["Marks"].fillna(0)
df["PreviousScore"] = df["PreviousScore"].fillna(df["PreviousScore"].mean())

# 5) Clean and encode ParentEdu
df["ParentEdu"] = df["ParentEdu"].replace({
    "Bachelor": "UniDegree",
    "BSc": "UniDegree",
    "University": "UniDegree",
    "Uni": "UniDegree",
    "HighSchool": "HighSchool",
    "School": "School",
    "NoEdu": "NoEdu"
})
parentedu_map = {"NoEdu": 0, "School": 1, "HighSchool": 2, "UniDegree": 3}
df["ParentEdu"] = df["ParentEdu"].map(parentedu_map)

# 6) Encode TestPrep
df["TestPrep"] = df["TestPrep"].map({"yes": 1, "no": 0})

# --- Regression starts here ---

# 7) Choose best features for FinalScore prediction
features = ["Attendance", "StudyHours", "Marks", "PreviousScore", "TestPrep"]

X = df[features]
y = df["FinalScore"]

# 8) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9) Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 10) Evaluate
y_pred = model.predict(X_test)

print("MAE (Mean Absolute Error):", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 11) Predict for a new student
new_student = pd.DataFrame([{
    "Attendance": 80,
    "StudyHours": 3,
    "Marks": 75,
    "PreviousScore": 70,
    "TestPrep": 1
}])

print("Predicted Final Score:", model.predict(new_student))
