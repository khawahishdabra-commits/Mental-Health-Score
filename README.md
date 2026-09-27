# Mansik-Santulan-Score

An end-to-end Machine Learning project that predicts a numerical mental-health-related score for students using social-media usage, study habits, sleep, physical activity, stress level, and demographic information.

> **Important:** This is an educational machine-learning system. The prediction is not a medical diagnosis, psychological assessment, or substitute for professional help.

## 🚀 Live Demo

- **Frontend:** https://mental-health-score-bay.vercel.app/
- **Backend API:** https://mental-health-score-2jbu.onrender.com/
- **GitHub:** https://github.com/khawahishdabra-commits/Mental-Health-Score

---

## 📌 Project Overview

Mansik-Santulan-Score is an end-to-end ML application built to practice real-world machine-learning engineering workflows.

The system:

- Cleans and preprocesses student data
- Performs reproducible feature preprocessing
- Trains and evaluates regression models
- Uses cross-validation for model validation
- Saves the complete preprocessing + model pipeline
- Serves predictions through a FastAPI REST API
- Provides a browser-based frontend
- Deploys the frontend and backend separately
- Connects the production frontend to the production ML API

The project focuses on **ML engineering practices rather than only model training**.

---

## 🏗️ System Architecture

```text
                    User
                     │
                     ▼
          ┌─────────────────────┐
          │   Vercel Frontend   │
          │  HTML + CSS + JS    │
          └──────────┬──────────┘
                     │
                     │ HTTPS POST /predict
                     ▼
          ┌─────────────────────┐
          │    Render Backend   │
          │      FastAPI        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Preprocessing       │
          │ Pipeline            │
          │                     │
          │ CountryGrouper      │
          │ Scaling             │
          │ Encoding            │
          └──────────┬──────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ RandomForestRegressor│
          │ n_estimators = 100   │
          │ max_features = sqrt  │
          └──────────┬───────────┘
                     │
                     ▼
             Mental Health Score
```

### Deployment

| Component | Platform |
|---|---|
| Frontend | Vercel |
| Backend API | Render |
| Source Code | GitHub |
| ML Model | scikit-learn |
| API | FastAPI |

---

## 📊 Dataset

Dataset:

`Student Social Media And Mental Health Impact.csv`

### Dataset Statistics

- Original rows: **5,000**
- Original columns: **13**
- Duplicate rows removed: **2**
- Final dataset: **4,998 rows**

### Features

| Feature | Description |
|---|---|
| Age | Student age |
| Gender | Student gender |
| Country | Student country |
| Academic_Level | Academic level |
| Most_Used_Platform | Most frequently used social platform |
| Purpose_Of_Use | Main purpose of social-media use |
| Avg_Daily_Usage_Hours | Average daily social-media usage |
| Daily_Unlocks | Number of daily device/app unlocks |
| Study_Hours | Daily study hours |
| Physical_Activity_Hours | Physical activity hours |
| Sleep_Hours_Per_Night | Average sleep duration |
| Stress_Level | Self-reported stress category |
| Mental_Health_Score | Regression target |

---

## ⚙️ Machine Learning Pipeline

The complete preprocessing and model are stored together inside a scikit-learn `Pipeline`.

This helps maintain consistency between training and production inference.

### Study Hours

The `Study_Hours` feature uses:

1. `log1p` transformation
2. Standard scaling

### Numerical Features

Numerical features are standardized using `StandardScaler`.

### Stress Level

Stress categories are ordinally encoded:

```text
Low → Medium → High → Very High
```

### Categorical Features

Categorical features use:

```text
OneHotEncoder(handle_unknown="ignore")
```

This allows the API to handle previously unseen categorical values without failing.

### Country Processing

A custom `CountryGrouper` transformer is used.

It:

1. Learns the top 10 countries from the training data.
2. Keeps those countries as separate categories.
3. Groups all remaining countries into `Other`.
4. Applies the same transformation during prediction.

This keeps country preprocessing inside the saved ML pipeline and reduces the risk of training-serving preprocessing mismatch.

---

## 🤖 Final Model

The selected model is:

```text
RandomForestRegressor
```

Configuration:

```text
n_estimators = 100
max_features = "sqrt"
random_state = 42
```

The complete preprocessing + model pipeline is saved as:

```text
Mental_Health_Model.pkl
```

---

## 📈 Model Evaluation

The final model was evaluated using a fixed **70/30 train-test split**.

### Test Set Results

| Model | Test R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.7398 | 0.5362 | 0.6760 |
| Random Forest | **0.8922** | **0.3252** | **0.4352** |

### 5-Fold Cross-Validation

Final Random Forest configuration:

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| R² | 0.8946 | 0.0076 |
| MAE | 0.3081 | 0.0020 |
| RMSE | 0.4138 | 0.0097 |

Cross-validation was used to check whether model performance was reasonably consistent across different training and validation splits.

---

## 🔬 Model Selection Experiments

Several controlled experiments were performed before selecting the final configuration.

| Experiment | Outcome |
|---|---|
| Linear Regression | Used as baseline |
| Random Forest | Strong improvement |
| `max_features="sqrt"` | Improved validation performance |
| Interaction features | Not selected |
| Log target transformation | Not selected |
| HistGradientBoosting | Not selected |
| Additional Random Forest tuning | No sufficient improvement over validated configuration |

The final configuration was selected using validation and test-set performance rather than simply choosing the most complex configuration.

---

## 🔎 Feature Importance

Permutation importance was used to investigate which features contributed most to predictive performance in the final model.

The strongest predictive signals included:

- Average daily usage hours
- Sleep hours
- Daily unlocks
- Most-used platform
- Country
- Purpose of use
- Study hours
- Physical activity
- Age
- Academic level
- Stress level
- Gender

> Feature importance describes predictive contribution within this model. It does **not** establish causation.

---

# 🚀 FastAPI Backend

The trained pipeline is exposed through a FastAPI REST API.

## Local Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the API Locally

```bash
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Information

```http
GET /model-info
```

Returns:

- Model type
- Number of estimators
- Expected input features

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "YouTube",
  "purpose_of_use": "Education",
  "avg_daily_usage_hours": 3.5,
  "daily_unlocks": 100,
  "study_hours": 5,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 7.5,
  "stress_level": "Medium"
}
```

Example response:

```json
{
  "predicted_mental_health_score": 7.8
}
```

---

# 🌐 Production Deployment

## Frontend

The frontend is deployed on Vercel.

**Live application:**  
https://mental-health-score-bay.vercel.app/

The frontend consists of:

- HTML
- CSS
- JavaScript

It sends prediction requests to the production FastAPI backend.

## Backend

The FastAPI backend is deployed on Render.

**Production API:**  
https://mental-health-score-2jbu.onrender.com/

The production environment uses pinned ML dependencies to maintain compatibility with the saved scikit-learn model.

Important dependencies include:

```text
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.1.3
joblib==1.4.2
```

The Python version is pinned using:

```text
.python-version
```

---

# 📁 Project Structure

```text
Mental-Health-Score/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── main.py
├── train.py
├── preprocessing.py
├── check_model.py
├── evaluate_model.py
├── test_prediction.py
│
├── Mental_Health_Model.pkl
├── Student Social Media And Mental Health Impact.csv
│
├── requirements.txt
├── .python-version
├── .gitattributes
├── .gitignore
└── README.md
```

Local experiment scripts used during model development are intentionally excluded from the Git repository.

---

# 🧪 Training the Model

Run:

```bash
python train.py
```

The training process:

1. Loads the dataset
2. Removes duplicate rows
3. Cleans physical activity values
4. Splits the dataset
5. Builds the preprocessing pipeline
6. Trains the Random Forest model
7. Evaluates the model
8. Performs cross-validation
9. Saves the complete pipeline

Output model:

```text
Mental_Health_Model.pkl
```

---

# 🔍 Testing the Saved Model

Check that the saved model can be loaded:

```bash
python check_model.py
```

Run a prediction test:

```bash
python test_prediction.py
```

Evaluate the model:

```bash
python evaluate_model.py
```

---

# ⚠️ Limitations

This project has several limitations:

1. The dataset is not a clinical mental-health dataset.
2. Predictions should not be interpreted as medical or psychological diagnoses.
3. The model identifies statistical patterns in the provided dataset.
4. Feature importance does not imply causality.
5. Performance on this dataset does not guarantee performance on real-world populations.
6. The dataset may not represent all students, countries, or demographic groups.
7. Independent external validation has not been performed.
8. The model should not be used to make medical or mental-health decisions.

---

# 🔮 Future Improvements

Potential improvements include:

- Independent external validation
- Automated model monitoring
- Prediction logging
- Automated testing
- Docker deployment
- CI/CD pipeline
- API authentication
- Model versioning
- Data drift monitoring
- Experiment tracking
- Explainable predictions
- Improved production configuration
- More robust frontend analytics

---

# 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- scikit-learn
- Joblib

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Vercel
- Render
- GitHub

---

# 📚 Project Purpose

This project was built as a practical exercise in:

- Machine Learning
- Regression
- Feature preprocessing
- Model evaluation
- Cross-validation
- Scikit-learn pipelines
- Custom transformers
- REST API development
- Frontend/API integration
- Cloud deployment
- Production dependency management
- Training-serving consistency

The primary goal is to demonstrate the transition from:

```text
Dataset
   ↓
Experimentation
   ↓
Model
   ↓
Pipeline
   ↓
API
   ↓
Frontend
   ↓
Cloud Deployment
```

rather than treating machine learning as only a notebook-based exercise.

---

# ⚠️ Disclaimer

This project is intended for educational and machine-learning engineering purposes only.

The predicted score should not be used to diagnose, treat, or make medical decisions about a person's mental health.

For concerns about mental health, users should consult a qualified healthcare professional.
