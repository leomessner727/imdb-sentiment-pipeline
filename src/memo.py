"""Step memoization pattern.

The output of a pipeline step is saved on disk, identified by the step name and
a hash of its parameters. When the same step is requested again with the same
parameters, the saved output is loaded instead of being recomputed. Changing a
parameter changes the hash, so the step is computed again automatically.
"""
import os

import joblib

from .config import CACHE_DIR
from .utils import get_logger, stable_hash

log = get_logger("memo")


def memoized_step(step_name, params, compute_fn):
    """Returns the output of a step, computing it only if it is not cached yet.

    step_name   name of the step, e.g. "split" or "features"
    params      all parameters that influence the output (JSON-serializable)
    compute_fn  function without arguments that computes the output
    """
    key = stable_hash({"step": step_name, "params": params})
    path = CACHE_DIR / f"{step_name}_{key}.joblib"

    if path.exists():
        log.info("step '%s' [%s]: cache hit, loading %s", step_name, key, path.name)
        return joblib.load(path)

    log.info("step '%s' [%s]: cache miss, computing", step_name, key)
    result = compute_fn()

    # Atomic write: save to a temporary file first, then rename it.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    joblib.dump(result, tmp_path)
    os.replace(tmp_path, path)
    log.info("step '%s' [%s]: saved %s", step_name, key, path.name)
    return result
