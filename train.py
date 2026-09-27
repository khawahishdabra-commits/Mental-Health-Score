import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from preprocessing import CountryGrouper


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("Student Social Media And Mental Health Impact.csv")

print("Original shape:", df.shape)


# -----------------------------
# 2. Basic data cleaning
# -----------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Physical activity cannot be negative
df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)

print("Shape after cleaning:", df.shape)


# -----------------------------
# 3. Define features and target
# -----------------------------

skewed_features = ["Study_Hours"]

numeric_features = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night"
]

ordinal_features = ["Stress_Level"]

categorical_features = [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
]

country_feature = ["Country"]

features = (
    skewed_features
    + numeric_features
    + ordinal_features
    + categorical_features
    + country_feature
)

X = df[features]
y = df["Mental_Health_Score"]


# -----------------------------
# 4. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# -----------------------------
# 5. Preprocessing pipelines
# -----------------------------

skewed_pipeline = Pipeline([
    ("log_transform", FunctionTransformer(np.log1p)),
    ("scale", StandardScaler())
])

numeric_pipeline = Pipeline([
    ("scale", StandardScaler())
])

ordinal_pipeline = Pipeline([
    (
        "encode",
        OrdinalEncoder(
            categories=[["Low", "Medium", "High", "Very High"]]
        )
    )
])

categorical_pipeline = Pipeline([
    (
        "encode",
        OneHotEncoder(handle_unknown="ignore")
    )
])

country_pipeline = Pipeline([
    ("group_country", CountryGrouper(n_top_countries=10)),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("skewed", skewed_pipeline, skewed_features),
    ("numeric", numeric_pipeline, numeric_features),
    ("ordinal", ordinal_pipeline, ordinal_features),
    ("categorical", categorical_pipeline, categorical_features),
    ("country", country_pipeline, country_feature)
])


# -----------------------------
# 6. Linear Regression baseline
# -----------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)


# -----------------------------
# 7. Final Random Forest
# -----------------------------
#
# Selected after controlled experiments
# and 5-fold cross-validation.
#
# n_estimators = 100
# max_features = "sqrt"
# random_state = 42
#
# This configuration performed better
# than the previous default Random Forest.
# -----------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_features="sqrt",
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_train_pred = random_forest_model.predict(X_train)
rf_test_pred = random_forest_model.predict(X_test)


# -----------------------------
# 8. Final Random Forest metrics
# -----------------------------

rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_test_rmse = np.sqrt(
    mean_squared_error(y_test, rf_test_pred)
)

print("\nFinal Random Forest:")
print("Parameters:")
print("n_estimators: 100")
print("max_features: sqrt")
print("random_state: 42")

print("\nMetrics:")
print("Train R2:", rf_train_r2)
print("Test R2:", rf_test_r2)
print("Test MAE:", rf_test_mae)
print("Test RMSE:", rf_test_rmse)


# -----------------------------
# 9. Save final production model
# -----------------------------

joblib.dump(
    random_forest_model,
    "Mental_Health_Model.pkl"
)

print("\nModel saved successfully:")
print("Mental_Health_Model.pkl")


# -----------------------------
# 10. Final model comparison
# -----------------------------

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "Train R2": [
        r2_score(y_train, linear_train_pred),
        rf_train_r2
    ],
    "Test R2": [
        r2_score(y_test, linear_test_pred),
        rf_test_r2
    ],
    "Test MAE": [
        mean_absolute_error(y_test, linear_test_pred),
        rf_test_mae
    ],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, linear_test_pred)),
        rf_test_rmse
    ]
})

print("\nFinal Model Comparison:")
print(results.to_string(index=False))