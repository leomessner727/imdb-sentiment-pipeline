"""Small demo: classifies a review with the best model chosen by the pipeline.

Usage: python -m src.predict "This movie was really great!"
"""
import sys

import joblib

from .config import FINAL_METRICS_FILE, MODELS_DIR
from .prepare import build_features, clean_text
from .utils import read_json


def main():
    if len(sys.argv) < 2:
        print('usage: python -m src.predict "text of a review"')
        sys.exit(1)
    if not FINAL_METRICS_FILE.exists():
        print("no trained model found: run the pipeline first")
        sys.exit(1)

    text = clean_text(" ".join(sys.argv[1:]))
    final = read_json(FINAL_METRICS_FILE)
    vectorizer = build_features(final["tfidf"])["vectorizer"]
    model = joblib.load(MODELS_DIR / f"{final['best_config']}.joblib")

    p_positive = model.predict_proba(vectorizer.transform([text]))[0][1]
    label = "positive" if p_positive >= 0.5 else "negative"
    print(f"{label} (probability positive = {p_positive:.3f}, model {final['best_config']})")


if __name__ == "__main__":
    main()
