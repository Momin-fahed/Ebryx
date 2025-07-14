import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved pipeline (make sure this file is in the same directory)
model = joblib.load("breast_cancer_decision_tree.joblib")

# List of features in the order expected by the model
FEATURES = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'radius_se', 'perimeter_se',
        'area_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

app = FastAPI()

class CancerInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    radius_se: float
    perimeter_se: float
    area_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

@app.post("/predict")
def predict(data: CancerInput):
    # Convert input to the correct order for the model
    X = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)
    pred = model.predict(X)[0]

    # Translate prediction
    diagnosis = "Malignant (Cancerous Tumor)" if pred == 1 else "Benign (Non-Cancerous Tumor)"

    return {
        "prediction": int(pred),
        "diagnosis": diagnosis
    }
