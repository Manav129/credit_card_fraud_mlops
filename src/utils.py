import os
import shutil
import joblib


def save_model(model, path):
    """
    Save the model to the given path.
    If a directory exists at the path, remove it first to avoid errors.
    """
    # Agar path directory hai toh use remove kar do (permission error avoid karne ke liye)
    if os.path.isdir(path):
        print(f"[Warning] Directory exists instead of file at {path}. Removing it.")
        shutil.rmtree(path)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model using joblib
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load the model from the given path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)
