from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.inference import get_prediction, Transaction

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://credit-card-fraud-frontend-seven.vercel.app"
    ],  # ✅ Your Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "API is running."}

@app.post("/predict")
def predict(transaction: Transaction):
    return get_prediction(transaction)
