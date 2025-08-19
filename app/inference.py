import joblib
import numpy as np
from pydantic import BaseModel

model = joblib.load("models/logistic_model.pkl")

class Transaction(BaseModel):
    features: list[float]  # 30 features including Time & Amount

def get_prediction(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
