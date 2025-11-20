import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
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
    "Bachelor":"UniDegree", "BSc":"UniDegree", "University":"UniDegree", "Uni":"UniDegree",
    "HighSchool":"HighSchool", "School":"School", "NoEdu":"NoEdu"
})
df["ParentEdu"] = df["ParentEdu"].map({"NoEdu":0, "School":1, "HighSchool":2, "UniDegree":3})

# Encode TestPrep
df["TestPrep"] = df["TestPrep"].map({"yes":1, "no":0})

# Features
features = ["Attendance", "StudyHours", "Marks", "PreviousScore", "TestPrep"]
X = df[features]
y = df["FinalScore"]

# Train RF model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_

# Plot
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

# Save to file inside images folder
plt.savefig("images/feature_importance.png")  # saves in `images/` folder
# plt.show()  # you can keep this too if you want to see it
