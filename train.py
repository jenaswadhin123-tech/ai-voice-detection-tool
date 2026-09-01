from pathlib import Path

import joblib
import numpy as np

from src.feature_extraction import extract_features
from src.model import train_model

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_audio"
MODEL_PATH = PROJECT_ROOT / "model.joblib"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_dataset():
    files = sorted(path for path in DATA_DIR.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS)
    labels = np.array([1 if "ai" in path.name.lower() else 0 for path in files])
    features = np.array([extract_features(path) for path in files])
    return files, features, labels


if __name__ == "__main__":
    files, features, labels = load_dataset()
    print(f"Found {len(files)} audio files.")
    if len(files) == 0:
        raise SystemExit("Add labeled audio files to data/sample_audio first.")

    model, accuracy = train_model(features, labels)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.3f}")
