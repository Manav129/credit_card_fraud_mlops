🛡️ Credit Card Fraud Detection (MLOps Project)
This project implements an end-to-end machine learning pipeline to detect fraudulent credit card transactions using Logistic Regression. It is designed with full MLOps integration using:

✅ MLflow for experiment tracking

✅ Docker + FastAPI for deployment

✅ GitHub Actions CI for auto-training, testing & linting

🔄 CD (Continuous Deployment) planned via Railway

📁 Project Structure

credit_card_fraud_mlops/
│
├── app/                         # FastAPI application
│   ├── main.py                  # FastAPI routes
│   └── inference.py             # Handles model predictions
│
├── data/                        # Dataset location
│   └── raw/                     # Downloaded from Kaggle
│
├── docker/                      # Docker-related files
│   └── Dockerfile               # Image for deployment
│
├── models/                      # Trained model + confusion matrix
│
├── notebooks/                   # EDA notebooks (optional)
│
├── src/                         # Core training & evaluation logic
│   ├── config.py                # Paths and global constants
│   ├── train.py                 # Model training script
│   ├── evaluate.py              # Model evaluation and logging
│   ├── data_loader.py           # Loads and preprocesses dataset
│   └── utils.py                 # Model save/load utilities
│
├── tests/                       # Unit tests
│   ├── test_api.py              # API testing
│   ├── test_data_loader.py      # Data loader testing
│   └── test_train.py            # Training logic testing
│
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # You're reading it now!
🚀 Features
Feature	Status
MLflow Tracking	✅ Done
GitHub Actions CI	✅ Done
Linting with Flake8	✅ Passed
Unit Testing with Pytest	✅ Passed
FastAPI + Docker Deploy	✅ Done (Local & Docker)
Dataset Auto Download	✅ (from Kaggle using GitHub Secret)
Continuous Deployment	🔄 Optional (Partially explored using Railway)

🧠 Model Info
Algorithm: Logistic Regression

Dataset: Credit Card Fraud Detection

Accuracy: ~99% (on imbalanced data)

Features: 30 anonymized transaction features

Output: Binary classification — Fraud (1) or Legitimate (0)

⚙️ MLOps Pipeline
✅ CI Workflow (via .github/workflows/ci.yml)
On push to main, the pipeline:

Downloads dataset from Kaggle

Trains and saves model

Logs confusion matrix

Runs evaluation and testing

Performs Flake8 lint check

✅ Docker + FastAPI
Containerized using Docker

Runs a FastAPI app with:

POST /predict → Accepts 30 features and returns prediction

GET / → Health check

Run locally using:


docker build -t credit-fraud-api -f docker/Dockerfile .
docker run -d -p 8000:8000 credit-fraud-api
🔐 Secrets & Credentials
KAGGLE_B64: Base64 encoded kaggle.json for dataset access

RAILWAY_TOKEN: (Optional) for CD via Railway

Secrets are safely managed using GitHub Actions.

🧪 Testing
Unit and integration tests are included:


pytest tests/
flake8 src tests
🧠 How to Use
Clone the repo

Install dependencies: pip install -r requirements.txt

Add Kaggle secret to GitHub

Push to main → Auto CI runs

Optional: Deploy using Docker or Railway

🔄 Future Work
Automate CD with Railway (Docker push + deploy on new commit)

Add monitoring/logging for live predictions

Use imbalanced-learn / SMOTE for better fraud classification

👥 Contributors
Manav — Lead Developer, MLOps Integration

Your teammates can now join by cloning:
👉 GitHub Repo: [https://github.com/<your-username>/<repo-name>]