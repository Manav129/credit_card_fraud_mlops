from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import traceback

from src.config import MODEL_PATH
from src.utils import load_model

# Initialize FastAPI app
app = FastAPI(
    title="💳 Credit Card Fraud Detection API",
    description="API for predicting credit card fraud using ML model",
    version="1.0.0",
)

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://credit-card-fraud-frontend-seven.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as err:
    print(f"🔥 Failed to load model: {err}")
    model = None


# Input data format
class InputData(BaseModel):
    features: List[float]  # Expecting 30 numerical features


# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        features = np.array(data.features).reshape(1, -1)

        if features.shape[1] != 30:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input size. Expected 30, got {features.shape[1]}"
            )

        prediction = model.predict(features)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0].tolist()
        else:
            prob = None

        return {"prediction": int(prediction[0]), "probability": prob}

    except HTTPException:
        # Let FastAPI handle expected errors
        raise
    except Exception as err:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed: {str(err)}")
