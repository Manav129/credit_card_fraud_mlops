# tests/test_api.py

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Credit Card Fraud Detection API"}

def test_predict_valid_input():
    payload = {
        "features": [0.1] * 30  # Dummy input matching PCA feature size
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_input():
    payload = {
        "features": [0.1] * 10  # Invalid input
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
