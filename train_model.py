import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_excel("data/indian-job-market-dataset-2025.xlsx")

print("Original dataset shape:", df.shape)


# ============================================================
# 2. SELECT REQUIRED COLUMNS
# ============================================================

salary_df = df[
    [
        "title",
        "tagsAndSkills",
        "experience",
        "location",
        "minimumSalary",
        "maximumSalary",
        "currency",
        "minimumExperience",
        "maximumExperience"
    ]
].copy()


# ============================================================
# 3. KEEP VALID INR SALARY RECORDS
# ============================================================

salary_df = salary_df[
    (salary_df["minimumSalary"] > 0) &
    (salary_df["maximumSalary"] > 0) &
    (salary_df["currency"] == "INR")
].copy()

print("Usable INR salary records:", salary_df.shape)


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

salary_df = salary_df.dropna(
    subset=[
        "title",
        "tagsAndSkills",
        "location",
        "minimumExperience",
        "maximumExperience"
    ]
).copy()


# ============================================================
# 5. CALCULATE AVERAGE SALARY
# ============================================================

salary_df["averageSalary"] = (
    salary_df["minimumSalary"] +
    salary_df["maximumSalary"]
) / 2


# ============================================================
# 6. REMOVE EXTREME SALARY OUTLIERS
# ============================================================

salary_limit = salary_df["averageSalary"].quantile(0.995)

salary_df = salary_df[
    salary_df["averageSalary"] <= salary_limit
].copy()

print(f"Salary limit used: ₹{salary_limit:,.0f}")
print("Records after outlier handling:", salary_df.shape)


# ============================================================
# 7. CALCULATE EXPERIENCE
# ============================================================

salary_df["experienceYears"] = (
    salary_df["minimumExperience"] +
    salary_df["maximumExperience"]
) / 2


# ============================================================
# 8. FEATURES
# ============================================================

X = salary_df[
    [
        "title",
        "tagsAndSkills",
        "location",
        "experienceYears"
    ]
]

# Target
y = salary_df["averageSalary"]

print("Features:", X.columns.tolist())
print("Target:", y.name)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 10. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "title",
            TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2
            ),
            "title"
        ),

        (
            "skills",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2
            ),
            "tagsAndSkills"
        ),

        (
            "location",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["location"]
        ),

        (
            "experience",
            "passthrough",
            ["experienceYears"]
        )
    ]
)


# ============================================================
# 11. MACHINE LEARNING MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            Ridge(alpha=10.0)
        )
    ]
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("Training model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Salary cannot be negative
y_pred = np.maximum(y_pred, 0)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation")
print("----------------")
print(f"MAE: ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


# ============================================================
# 15. SALARY DISTRIBUTION
# ============================================================

print("\nSalary percentiles")
print("------------------")

print(
    salary_df["averageSalary"].quantile(
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.995,
            1.00
        ]
    )
)


# ============================================================
# 16. SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame(
    {
        "ActualSalary": y_test.values,
        "PredictedSalary": y_pred
    }
)

results["Error"] = (
    results["PredictedSalary"] -
    results["ActualSalary"]
)

results["AbsoluteError"] = (
    results["Error"].abs()
)


print("\nSample predictions")
print("------------------")

print(
    results.head(10)
)


# ============================================================
# 17. LARGEST ERRORS
# ============================================================

print("\nLargest prediction errors")
print("-------------------------")

print(
    results.sort_values(
        "AbsoluteError",
        ascending=False
    ).head(10)
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "model/salary_prediction_model.pkl"
)

print("\nModel saved successfully!")
print("File: model/salary_prediction_model.pkl")