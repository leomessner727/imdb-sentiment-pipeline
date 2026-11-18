"""Step 2 of the pipeline (fan-out): trains ONE model configuration.

Several workers run at the same time as separate processes (pipeline.py),
each with a different configuration. Every worker writes its validation
metrics to data/results/<id>.json and its model to data/models/<id>.joblib.

Usage: python -m src.worker --config w1
"""
import argparse
import os
import socket
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .config import MODELS_DIR, RESULTS_DIR, SEED, TFIDF_VARIANTS, get_config
from .prepare import build_features
from .utils import get_logger, read_json, set_seeds, stable_hash, write_json_atomic

log = get_logger("worker")


def train_config(config_id):
    """Trains the logistic regression of one configuration and returns its metrics."""
    set_seeds()
    cfg = get_config(config_id)
    run_key = stable_hash({"config": cfg, "tfidf": TFIDF_VARIANTS[cfg["tfidf"]], "seed": SEED})
    result_path = RESULTS_DIR / f"{cfg['id']}.json"
    model_path = MODELS_DIR / f"{cfg['id']}.joblib"

    # Step memoization at worker level: if this exact configuration has
    # already been trained, the saved result is reused.
    if result_path.exists() and model_path.exists():
        previous = read_json(result_path)
        if previous.get("run_key") == run_key:
            log.info("config %s: already trained (run_key %s), skipping", cfg["id"], run_key)
            return previous

    # Features of this variant: normally a cache hit, prepared in step 1
    data = build_features(cfg["tfidf"])

    log.info("config %s: training (tfidf=%s, C=%s) in process %d",
             cfg["id"], cfg["tfidf"], cfg["C"], os.getpid())
    model = LogisticRegression(C=cfg["C"], solver="liblinear", max_iter=1000, random_state=SEED)
    start = time.time()
    model.fit(data["X_train"], data["y_train"])
    train_seconds = time.time() - start

    # Evaluation on the validation set (the test set is used only once, in step 3)
    y_pred = model.predict(data["X_val"])
    result = {
        "config_id": cfg["id"],
        "tfidf": cfg["tfidf"],
        "C": cfg["C"],
        "val_accuracy": float(accuracy_score(data["y_val"], y_pred)),
        "val_f1": float(f1_score(data["y_val"], y_pred)),
        "train_seconds": round(train_seconds, 2),
        "host": socket.gethostname(),
        "pid": os.getpid(),  # process id: every worker runs in its own process
        "run_key": run_key,
    }

    # The model is saved before the result: an existing result file
    # therefore always means that the model is complete.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    write_json_atomic(result_path, result)
    log.info("config %s: val accuracy %.4f, val F1 %.4f, %.1f s",
             cfg["id"], result["val_accuracy"], result["val_f1"], train_seconds)
    return result


def main():
    parser = argparse.ArgumentParser(description="Trains one model configuration (fan-out worker).")
    parser.add_argument("--config", required=True, help="configuration id, e.g. w1")
    args = parser.parse_args()
    train_config(args.config)


if __name__ == "__main__":
    main()
