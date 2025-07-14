from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

cluster_labels = {
    0: "Cautious High-Income Adults",
    1: "Affluent Young Big Spenders",
    2: "Impulsive Budget Youth",
    3: "Moderate-Earning Young Adults",
    4: "Frugal Low-Income Seniors",
}


# Initialize FastAPI app
app = FastAPI(
    title="Customer Clustering API",
    description="Predicts the customer segment (cluster) using age, income, and spending score.",
    version="1.0"
)

# Load pipeline
pipeline = joblib.load("customer_cluster_pipeline.pkl")

# Input schema
class CustomerInput(BaseModel):
    age: float
    income: float
    score: float

# Route: Home
@app.get("/")
def home():
    return {"message": "Customer Clustering API is running!"}

# Route: Predict cluster
@app.post("/predict")
def predict_cluster(data: CustomerInput):  # Fixed here
    input_data = np.array([[data.age, data.income, data.score]])
    cluster = pipeline.predict(input_data)[0]
    label = cluster_labels.get(cluster, f"Cluster {cluster}")
    return {
        "cluster": int(cluster),
        "label": label
    }
