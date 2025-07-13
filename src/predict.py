# src/predict.py

import joblib
import numpy as np

MODEL_PATH = "models/logistic_model.pkl"

def load_model(path: str = MODEL_PATH):
    return joblib.load(path)

def make_prediction(model, input_data: list):
    """
    Predict class for a list of input features.

    Args:
        model: Trained model object
        input_data: List of feature values

    Returns:
        int: 0 (not fraud) or 1 (fraud)
    """
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return int(prediction[0])
