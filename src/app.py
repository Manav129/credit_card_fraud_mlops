from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import traceback

from src.config import MODEL_PATH
from src.utils import load_model

# Initialize FastAPI app
app = FastAPI(
    title="💳 Credit Card Fraud Detection API",
    description="API for predicting credit card fraud using ML model",
    version="1.0.0",
)

# Enable CORS for your specific frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your live Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"🔥 Failed to load model: {e}")
    model = None


# Input data format
class InputData(BaseModel):
    features: list[float]  # Expecting 30 numerical features


# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Unavailable.")

    try:
        features = np.array(data.features).reshape(1, -1)

        if features.shape[1] != 30:
            raise HTTPException(
                status_code=400,
                detail=f". Got {features.shape[1]}."
            )

        # Make prediction
        prediction = model.predict(features)

        # Optional: probability if model supports predict_proba
        prob = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else None

        return {"prediction": int(prediction[0]), "probability": prob}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
