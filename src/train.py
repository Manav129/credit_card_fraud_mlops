import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    CONF_MATRIX_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    MAX_ITER
)


def load_data():
    """Load dataset from CSV."""
    return pd.read_csv(DATA_PATH)


def preprocess_data(df):
    """Split features and target, then train-test split."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix image."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    clf = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    plot_confusion_matrix(y_test, y_pred, CONF_MATRIX_PATH)
    print(f"Confusion matrix saved at {CONF_MATRIX_PATH}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train()
