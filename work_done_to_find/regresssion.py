from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource]

# 1) Data from your dataset
X = [
    [96, 98],
    [26, 12],
    [58, 48],
    [87, 79],
    [33, 22]
]

y = [1, 0, 0, 1, 0]   # 1 = pass, 0 = fail

# 2) Create the model
model = LogisticRegression()

# 3) Train the model on your data
model.fit(X, y)

# 4) Try a new student
new_student = [[70, 50]]   # attendance, marks
prediction = model.predict(new_student)

print("Prediction for (70, 50):", prediction)
