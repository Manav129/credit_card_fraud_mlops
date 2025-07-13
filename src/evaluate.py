import joblib
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_data, preprocess_data
from src.config import MODEL_PATH, CONF_MATRIX_PATH

def evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH)
    print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

    with mlflow.start_run():
        mlflow.log_artifact(CONF_MATRIX_PATH)

if __name__ == "__main__":
    evaluate()
