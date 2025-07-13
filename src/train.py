import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_data, preprocess_data
from src.utils import save_model
from src.config import MODEL_PATH

mlflow.set_experiment("credit-card-fraud-detection")

def train():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    with mlflow.start_run():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        save_model(clf, MODEL_PATH)
        mlflow.sklearn.log_model(clf, "model")

if __name__ == "__main__":
    train()
