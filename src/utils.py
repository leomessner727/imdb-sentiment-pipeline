"""Small helpers shared by all modules: seeds, logging, hashing, JSON files."""
import hashlib
import json
import logging
import os
import random

import numpy as np

from .config import SEED


def set_seeds(seed=SEED):
    """Fixes the random seeds of Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name):
    """Logger that prints time, module name and process id (useful with parallel workers)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s pid=%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def stable_hash(obj):
    """Short hash of a JSON-serializable object: same content, same hash."""
    text = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def write_json_atomic(path, data):
    """Writes a JSON file atomically: first to a temporary file, then renamed.
    Other processes never read a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_json(path):
    """Reads a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
