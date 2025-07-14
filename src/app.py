from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import traceback

from src.config import MODEL_PATH
from src.utils import load_model


app = FastAPI()


# Load the trained model
model = load_model(MODEL_PATH)


# Define the input data format
class InputData(BaseModel):
    features: list[float]


# Root endpoint
@app.get("/")
def home():
    return {"message": "FastAPI server is running!"}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("🔥 Error during prediction:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
