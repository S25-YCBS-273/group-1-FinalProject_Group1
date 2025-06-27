# FinalProject_Group1
 Introduction to Practical Machine Learning - Final Project 


# Set up your environment
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
jupyter notebook \
    --notebook-dir="." \
    --ip=0.0.0.0 --port=3225
```

```
fastapi dev main.py
```


# Preprocessing steps include
KNN imputation for missing numeric values (k=5)

One-hot encoding for nominal variables

Log transformation on MonthlyIncome to reduce skewness

Experience binning (New, Mid, Senior, Expert)

Feature selection using SelectKBest + RFE with XGBoost estimator

Train models

#odels used:

RandomForestRegressor

GradientBoostingRegressor

XGBRegressor

Hyperparameter tuning was done via GridSearchCV for each model.

Performance metric: Root Mean Squared Error (RMSE)

Save models


# Example: 
```
import joblib
joblib.dump(model, "xgboost_model.pkl")
```
Endpoint Details for Demo
To serve the trained model and run salary predictions via API:

Use Flask (or FastAPI) to build a local demo endpoint
from flask import Flask, request, jsonify

```
mport joblib
import pandas as pd
```

```
pp = Flask(__name__)
model = joblib.load("xgboost_model.pkl")
```

```
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'predicted_salary': float(prediction[0])})
```



```
if __name__ == '__main__':
    app.run(debug=True)
Demo Example (using curl):
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"Age": 30, "EducationLevel_encoded": 2, "YearsAtCompany": 5, ...}'

