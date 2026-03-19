"""
Fake News Detector — app.py
============================
Loads the best trained model and provides a clean predict() function
plus a simple CLI for testing.

Usage:
    python app.py                          # interactive CLI
    python app.py --text "your headline"  # single prediction
"""

import re
import os
import argparse
import joblib
import numpy as np

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
LABEL_MAP  = {0: "FAKE", 1: "REAL"}

# ── Load model once at import time ────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully \n")


# ── Text preprocessing (must match notebook) ──────────────────
def preprocess_text(text: str) -> str:
    """Same cleaning applied during training."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()            # normalise spaces
    return text


# ── Core prediction function ──────────────────────────────────
def predict(text: str) -> dict:
    """
    Predict whether a news article is Fake or Real.

    Parameters
    ----------
    text : str
        Raw article text or headline.

    Returns
    -------
    dict with keys:
        label       : "FAKE" or "REAL"
        label_int   : 0 (Fake) or 1 (Real)
        confidence  : float in [0, 1]  — confidence in the predicted class
    """
    cleaned = preprocess_text(text)

    # Predict label
    label_int = int(model.predict([cleaned])[0])

    # Confidence — use predict_proba if available, else decision_function
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba([cleaned])[0]
        confidence = float(proba[label_int])
    elif hasattr(model, "decision_function"):
        score      = model.decision_function([cleaned])[0]
        # Convert raw score to a 0-1 confidence via sigmoid
        confidence = float(1 / (1 + np.exp(-abs(score))))
    else:
        confidence = None

    return {
        "label":      LABEL_MAP[label_int],
        "label_int":  label_int,
        "confidence": round(confidence, 4) if confidence is not None else "N/A",
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """Predict for a list of texts."""
    return [predict(t) for t in texts]


# ── CLI ───────────────────────────────────────────────────────
def _print_result(text: str, result: dict):
    icon  = "REAL" if result["label"] == "REAL" else "FAKE"
    conf  = f"{result['confidence']*100:.1f}%" if isinstance(result["confidence"], float) else result["confidence"]
    print(f"\n{'─'*60}")
    print(f"  Text       : {text[:80]}{'...' if len(text)>80 else ''}")
    print(f"  Verdict    : {icon}")
    print(f"  Confidence : {conf}")
    print(f"{'─'*60}\n")


def interactive_cli():
    print("=" * 60)
    print("  FAKE NEWS DETECTOR — Interactive Mode")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    while True:
        text = input("\nPaste news text or headline:\n> ").strip()
        if text.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break
        result = predict(text)
        _print_result(text, result)


def main():
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument("--text", type=str, default=None,
                        help="News article text to classify")
    args = parser.parse_args()

    if args.text:
        result = predict(args.text)
        _print_result(args.text, result)
    else:
        interactive_cli()


if __name__ == "__main__":
    main()