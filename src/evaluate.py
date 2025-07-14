import os
import joblib
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_data, preprocess_data
from src.config import MODEL_PATH, CONF_MATRIX_PATH


def evaluate():
    # Load data and preprocess
    df = load_data()
    _, X_test, _, y_test, _ = preprocess_data(df)

    # Load trained model
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Generate predictions and evaluation metrics
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH)
    print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

    # Log confusion matrix as MLflow artifact
    with mlflow.start_run():
        mlflow.log_artifact(CONF_MATRIX_PATH)


if __name__ == "__main__":
    evaluate()
