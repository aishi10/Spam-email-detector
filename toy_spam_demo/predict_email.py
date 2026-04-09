from __future__ import annotations

import argparse
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"


def predict(text: str) -> tuple[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train the model first with train_spam_classifier.py."
        )

    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict([text])[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        confidence = float(max(probs))
    return ("SPAM" if prediction == 1 else "HAM", confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether text is spam or ham.")
    parser.add_argument("text", help="Message text to classify")
    args = parser.parse_args()

    label, confidence = predict(args.text)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
