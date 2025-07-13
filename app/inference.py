# app/inference.py

from fastapi import HTTPException
from pydantic import BaseModel
from src.predict import load_model, make_prediction

# Load the model once on startup
model = load_model()

class Transaction(BaseModel):
    features: list[float]  # 30 features from PCA-transformed dataset

def get_prediction(data: Transaction):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Input must have 30 features.")

    prediction = make_prediction(model, data.features)
    return {"prediction": prediction}
