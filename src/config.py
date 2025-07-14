import os

# Dynamically detect BASE_DIR from GitHub Actions or local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(BASE_DIR, ".git")) and BASE_DIR != "/":
    BASE_DIR = os.path.dirname(BASE_DIR)

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# Model and output paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CONF_MATRIX_PATH = CONFUSION_MATRIX_PATH  # alias if needed

# MLflow settings
MLFLOW_TRACKING_URI = "file://" + os.path.join(BASE_DIR, "mlruns")
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-detection"

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
