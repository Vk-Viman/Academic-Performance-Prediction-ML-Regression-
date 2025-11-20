import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create a simple dataset (we'll replace this with CSV later)
data = {
    "Attendance": [85, 60, 92, 55, 70, 45, 95, 80],
    "StudyHours": [3, 1, 4, 2, 3, 4, 1, 0],
    "Marks": [78, 40, 88, 38, 65, 65, 90, 30],
    "Pass": [1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

print(df)

# Split into features (X) and label (y)
X = df[["Attendance", "StudyHours", "Marks"]]
y = df["Pass"]

# Train/test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Try a prediction
new_student = pd.DataFrame(
    [[70, 3, 55]],
    columns=["Attendance", "StudyHours", "Marks"]
)

print("Prediction for (70, 3, 55):", model.predict(new_student))


