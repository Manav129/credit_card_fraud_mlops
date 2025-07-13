import os

# Base paths
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Data path
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CONF_MATRIX_PATH = CONFUSION_MATRIX_PATH  # Optional alias

# MLflow
MLFLOW_TRACKING_URI = "file://" + os.path.join(BASE_DIR, "mlruns")
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-detection"

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000
