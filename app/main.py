from fastapi import FastAPI
from app.inference import get_prediction, Transaction

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running."}

@app.post("/predict")
def predict(transaction: Transaction):
    return get_prediction(transaction)
