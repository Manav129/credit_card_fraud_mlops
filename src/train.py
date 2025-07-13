import os
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_data, preprocess_data
from src.utils import save_model
from src.config import MODEL_PATH, MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def train():
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Ensure model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save and log model
        save_model(clf, MODEL_PATH)
        mlflow.sklearn.log_model(clf, "model")


if __name__ == "__main__":
    train()
