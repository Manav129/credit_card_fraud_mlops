from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import traceback

from src.config import MODEL_PATH
from src.utils import load_model

app = FastAPI()

# Enable CORS (needed for frontend-backend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Keep * for first deploy; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load_model(MODEL_PATH)


# Define the input data format
class InputData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    """Root endpoint for testing."""
    return {"message": "Welcome to the Credit Card Fraud Detection API"}


@app.post("/predict")
def predict(data: InputData):
    """Prediction endpoint."""
    try:
        features = np.array(data.features).reshape(1, -1)

        if features.shape[1] != 30:
            raise HTTPException(
                status_code=400,
                detail="Input must contain exactly 30 features."
            )

        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}

    except HTTPException:
        raise

    except Exception as e:
        print("🔥 Error during prediction:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
