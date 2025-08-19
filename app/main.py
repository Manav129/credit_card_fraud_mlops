from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app import predict, InputData  # ✅ reuse schema & logic

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://credit-card-fraud-frontend-seven.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running."}

@app.post("/predict")
def predict_endpoint(data: InputData):   # ✅ match frontend payload
    return predict(data)
