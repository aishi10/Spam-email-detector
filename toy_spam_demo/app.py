from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )
    return joblib.load(MODEL_PATH)


def predict_message(text: str):
    model = load_model()
    prediction_num = int(model.predict([text])[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        confidence = float(max(probs))

    return {
        "label": "SPAM" if prediction_num == 1 else "HAM",
        "confidence": confidence,
        "text": text,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    result = None
    error = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            error = "Please type a message to classify."
        else:
            try:
                result = predict_message(message)
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        message=message,
        result=result,
        error=error,
        model_ready=MODEL_PATH.exists(),
        model_path=str(MODEL_PATH),
        confusion_matrix_path=str(CONFUSION_MATRIX_PATH),
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
