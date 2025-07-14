import os

# Project base directory (two levels up from this file's location)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data file path
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

# Model directory and file path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")

# Confusion matrix image path
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CONF_MATRIX_PATH = CONFUSION_MATRIX_PATH  # alias for confusion matrix if needed

# MLflow tracking URI - local file-based tracking inside project
MLFLOW_TRACKING_URI = "file://" + os.path.join(BASE_DIR, "mlruns")

# MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-detection"

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000

# Ensure models directory exists before training or saving
os.makedirs(MODEL_DIR, exist_ok=True)
