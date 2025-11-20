# Academic Performance Prediction (ML)

Repository reorganized into a clearer structure.

# Academic Performance Prediction — Machine Learning Project

This project develops machine learning models to predict student academic performance using a realistically messy educational dataset. It includes a complete data-cleaning pipeline, feature engineering for both numeric and categorical variables, and implementations of classification and regression models to predict pass/fail outcomes and final exam scores.

## 1. Overview

- Dataset: synthetic but realistic students dataset with 150+ records and 10 columns.
- Targets: `Pass` (classification) and `FinalScore` (regression).
- Models: Logistic Regression for classification; Linear Regression and Random Forest Regressor for regression.

## 2. Dataset Description

The dataset contains academic, behavioral, and demographic attributes such as `Attendance`, `StudyHours`, `Marks`, `ParentEdu`, `Absences`, `FamilyIncome`, `PreviousScore`, and `TestPrep`.

Data issues addressed:
- Missing value placeholders (`"?"`) converted to NaN.
- Non-numeric formatting in several columns handled via `pd.to_numeric(..., errors='coerce')`.
- Inconsistent categorical labels standardized (e.g. `Bachelor`, `BSc`, `Uni`, `University` → `UniDegree`).

## 3. Data Cleaning & Preprocessing

3.1 Handling Missing and Invalid Values
- Replace `"?"` with `NaN`.
- Convert numeric columns using `pd.to_numeric(..., errors='coerce')`.
- Fill missing numeric values:
  - `Attendance` → `0`
  - `StudyHours` → column mean
  - `Marks` → `0`
  - `PreviousScore` → column mean

3.2 Standardizing Categorical Labels
- `Bachelor`, `BSc`, `University`, `Uni` → `UniDegree`
- `HighSchool` → `HighSchool`
- `School` → `School`
- `NoEdu` → `NoEdu`

3.3 Encoding Categorical Variables
- `ParentEdu` (ordered encoding): `NoEdu = 0`, `School = 1`, `HighSchool = 2`, `UniDegree = 3`
- `TestPrep` (binary): `yes = 1`, `no = 0`

3.4 Feature Selection
- For classification (`Pass`): `Attendance`, `StudyHours`, `Marks`, `ParentEdu`, `Absences`, `FamilyIncome`, `PreviousScore`, `TestPrep`.
- For regression (`FinalScore`): `Attendance`, `StudyHours`, `Marks`, `PreviousScore`, `TestPrep`.

## 4. Model Training & Evaluation

### 4.1 Classification — Predicting Pass/Fail
- Model: Logistic Regression
- Reported accuracy on this dataset: `1.00` (perfect accuracy due to strong patterns and the small synthetic dataset).

### 4.2 Regression — Predicting FinalScore

Two models were trained and compared:
- Linear Regression
  - MAE: ~1.76
  - R²: ~0.991
- Random Forest Regressor
  - MAE: ~2.84
  - R²: ~0.966

Interpretation: Linear Regression performed best on this dataset (strong linear signals). Random Forest captured non-linear patterns but gave slightly lower accuracy here.

### 4.3 Model Comparison Summary
Task — Best Model — Reason
- Pass/Fail — Logistic Regression — Strong linear separation
- FinalScore — Linear Regression — Highest accuracy
- Complex modeling — Random Forest — Captures non-linear patterns

## 5. Feature Importance Analysis

5.1 Linear Regression Coefficients (example values)
- `Attendance`: -0.02798
- `StudyHours`: -0.07545
- `Marks`: 0.06089
- `PreviousScore`: 1.01671
- `TestPrep`: 1.04182

Interpretation: `PreviousScore` and `TestPrep` are strong linear predictors.

5.2 Random Forest Feature Importance (example values)
- `Attendance`: ~0.04
- `StudyHours`: ~0.005
- `Marks`: ~0.59
- `PreviousScore`: ~0.36
- `TestPrep`: ~0.0003

Interpretation: `Marks` dominates tree-based importance; `PreviousScore` is also important.

5.3 Feature Importance Plot
- Generated via `analysis/rf_feature_plot.py` and saved to `images/feature_importance.png`.

## 6. How to Run This Project

1. Clone the repository

```bash
git clone https://github.com/Vk-Viman/Academic-Performance-Prediction-ML-Regression-.git
cd Academic-Performance-Prediction-ML-Regression-
```

2. Create a virtual environment (Windows PowerShell example)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

3. Run model scripts (examples)

```powershell
python models\\student_model.py
python models\\student_regression_linear.py
python models\\student_regression_rf.py
python analysis\\rf_feature_plot.py
```

> Note: The scripts read the dataset from `data/students.csv`.

## 7. Project Structure

Academic-Performance-Prediction-ML/
│
├── data/
│   └── `students.csv`
│
├── models/
│   ├── `student_model.py`
│   ├── `student_regression_linear.py`
│   ├── `student_regression_rf.py`
│
├── analysis/
│   ├── `linear_feature_importance.py`
│   ├── `rf_feature_importance.py`
│   ├── `rf_feature_plot.py`
│
├── images/
│   └── `feature_importance.png`
│
├── README.md
├── requirements.txt
└── .gitignore

## 8. Future Improvements

- Hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`).
- Add more advanced models (XGBoost, LightGBM, neural networks).
- Create a FastAPI backend for real-time predictions.
- Build an interactive Streamlit dashboard for exploration.
- Add cross-validation for model stability and robustness.
- Expand the dataset to improve generalization.

---

If you want, I can also:
- Run one of the model scripts and paste the output here.
- Commit and push this README update (I can do that for you).
- Convert the scripts into importable functions and add a small CLI or notebook.
