from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

from src.config import MODEL_PATH
from src.utils import load_model

app = FastAPI()

model = load_model(MODEL_PATH)


class InputData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "FastAPI server is running!"}


@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
