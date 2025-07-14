import os
import shutil
import joblib


def save_model(model, path):
    """
    Save the model to the given path.

    If a directory exists at the path, remove it first to avoid conflicts.
    """
    if os.path.isdir(path):
        print(f"[Warning] Directory exists instead of file at {path}. Removing it.")
        shutil.rmtree(path)

    # Ensure parent directory exists
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)

    # Save model
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Load the model from the given path.

    Raises an error if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)
