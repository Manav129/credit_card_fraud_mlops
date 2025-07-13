# app/main.py

from fastapi import FastAPI
from app.inference import get_prediction, Transaction

app = FastAPI(title="Credit Card Fraud Detection API")

@app.get("/")
def root():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}

@app.post("/predict")
def predict(data: Transaction):
    return get_prediction(data)
