import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Load data
# adjusted to new data folder
df = pd.read_csv("data/students.csv")
df.replace("?", np.nan, inplace=True)

# Convert numerics
numeric_cols = ["Attendance", "StudyHours", "Marks", "PreviousScore", "FinalScore"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaNs
df["Attendance"] = df["Attendance"].fillna(0)
df["StudyHours"] = df["StudyHours"].fillna(df["StudyHours"].mean())
df["Marks"] = df["Marks"].fillna(0)
df["PreviousScore"] = df["PreviousScore"].fillna(df["PreviousScore"].mean())

# Clean ParentEdu
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

# Encode TestPrep
df["TestPrep"] = df["TestPrep"].map({"yes": 1, "no": 0})

# --- Regression model ---
features = ["Attendance", "StudyHours", "Marks", "PreviousScore", "TestPrep"]
X = df[features]
y = df["FinalScore"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("RF MAE:", mean_absolute_error(y_test, y_pred))
print("RF R2 Score:", r2_score(y_test, y_pred))

# Prediction on new student
new_student = pd.DataFrame([{
    "Attendance": 80,
    "StudyHours": 3,
    "Marks": 75,
    "PreviousScore": 70,
    "TestPrep": 1
}])

print("RF Predicted Final Score:", model.predict(new_student))
