from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource]

# Your dataset
X = [
    [96, 98],
    [26, 12],
    [58, 48],
    [87, 79],
    [33, 22]
]

y = [1, 0, 0, 1, 0]   # 1=pass, 0=fail

# Create and train model
model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully!")

# Predict for a new student
prediction = model.predict([[70, 50]])
print("Prediction for (70, 50):", prediction)

