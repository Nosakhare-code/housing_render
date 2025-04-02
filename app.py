from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import dill
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load trained model (replace with actual model path)
with open("grid_search.pkl", "rb") as f:
    model = dill.load(f)  # model contains a pipeline

# Define input schema
class ModelInput(BaseModel):
    float_features: List[float]  # 11 float features
    int_features: List[int]  # 28 integer features
    cat_features: List[str]  # 37 categorical features

# Preprocessing function
def preprocess_input(input_data: ModelInput):
    # Combine all features into a single list
    features = input_data.float_features + input_data.int_features + input_data.cat_features

    # Ensure the column names match the trained model's feature names
    feature_names = model.feature_names_in_  # This retrieves feature names from sklearn pipeline

    # Convert to DataFrame
    df = pd.DataFrame([features], columns=feature_names)
    
    return df

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: ModelInput):
    # Preprocess input
    data = preprocess_input(input_data)

    # Make prediction
    prediction = model.predict(data)
    
    return {"prediction": prediction.tolist()}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "ML Model API is running!"}
