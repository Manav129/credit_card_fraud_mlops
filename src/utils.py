import os
import joblib


def save_model(model, path):
    """
    Save the model to the given path.
    Ensure the parent directory exists.
    """
    if os.path.isdir(path):
        print(
            "[Warning] Directory exists instead of file at "
            f"{path}. Removing it."
        )

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
