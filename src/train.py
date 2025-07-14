import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Paths
DATA_PATH = os.path.join("data", "raw", "creditcard.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
CONF_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load dataset from CSV."""
    return pd.read_csv(DATA_PATH)


def preprocess_data(df):
    """Split features and target, then train-test split."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    plot_confusion_matrix(y_test, y_pred, CONF_MATRIX_PATH)
    print(f"Confusion matrix saved at {CONF_MATRIX_PATH}")

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train()
