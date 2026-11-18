"""Central configuration of the pipeline: seed, folders, data and model settings."""
import os
from pathlib import Path

# Fixed random seed for reproducibility (data split and model training)
SEED = 42

# Folders. DATA_DIR can be overridden with an environment variable,
# e.g. to keep the data of different runs apart.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"          # outputs of memoized steps
RESULTS_DIR = DATA_DIR / "results"      # one JSON file per trained configuration
MODELS_DIR = DATA_DIR / "models"        # one trained model per configuration
FINAL_METRICS_FILE = DATA_DIR / "final_metrics.json"

# Dataset: IMDB movie reviews from Hugging Face
# (25,000 training and 25,000 test reviews, label 0 = negative, 1 = positive)
DATASET_NAME = "stanfordnlp/imdb"
VAL_SIZE = 0.2  # 20% of the training reviews are kept apart as validation set

# TF-IDF feature variants. Each variant is computed once and then reused
# by every configuration that needs it (step memoization).
TFIDF_VARIANTS = {
    "uni": {"ngram_range": [1, 1], "min_df": 2, "max_features": 50000, "sublinear_tf": True},
    "bi": {"ngram_range": [1, 2], "min_df": 2, "max_features": 100000, "sublinear_tf": True},
}

# Model configurations trained in parallel (fan-out): one worker per configuration.
# C is the inverse regularization strength of the logistic regression.
CONFIGS = [
    {"id": "w1", "tfidf": "uni", "C": 1.0},
    {"id": "w2", "tfidf": "uni", "C": 4.0},
    {"id": "w3", "tfidf": "bi", "C": 1.0},
    {"id": "w4", "tfidf": "bi", "C": 4.0},
]

# Number of parallel worker processes started by pipeline.py
N_WORKERS = len(CONFIGS)


def get_config(config_id):
    """Returns the configuration with the given id, e.g. "w1"."""
    for cfg in CONFIGS:
        if cfg["id"] == config_id:
            return cfg
    valid = ", ".join(cfg["id"] for cfg in CONFIGS)
    raise ValueError(f"unknown configuration '{config_id}', valid ids: {valid}")
