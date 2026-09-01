import numpy as np
from src.feature_extraction import extract_features

def predict(model, file_path):
    """
    Predict if audio is AI-generated or Human.
    
    Args:
        model: Trained classifier
        file_path: Path to audio file
    
    Returns:
        str: "AI" or "Human"
    """
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    pred = model.predict(features)[0]
    return "AI" if pred == 1 else "Human"

def predict_with_confidence(model, file_path):
    """Return the predicted label and model probability for an audio file."""
    features = extract_features(file_path).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    ai_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    label = "AI" if ai_probability >= 0.5 else "Human"
    confidence = ai_probability if label == "AI" else 1 - ai_probability
    return label, confidence
