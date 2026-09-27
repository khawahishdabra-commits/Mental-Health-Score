import joblib
import pandas as pd


# Load the production model
model = joblib.load("Mental_Health_Model.pkl")


# Raw input data
input_data = pd.DataFrame([{
    "Age": 21,
    "Gender": "Male",
    "Country": "India",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "YouTube",
    "Purpose_Of_Use": "Education",
    "Avg_Daily_Usage_Hours": 3.5,
    "Daily_Unlocks": 100,
    "Study_Hours": 5.0,
    "Physical_Activity_Hours": 1.5,
    "Sleep_Hours_Per_Night": 7.5,
    "Stress_Level": "Medium"
}])


# The pipeline handles preprocessing automatically.
prediction = model.predict(input_data)[0]


print("Input:")
print(input_data.to_string(index=False))

print("\nPredicted Mental Health Score:", round(float(prediction), 2))