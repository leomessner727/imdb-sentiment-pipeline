"""Main module: runs the whole pipeline.

  1) prepare   data ingestion and TF-IDF features (memoized)
  2) fan-out   one worker process per configuration, all in parallel
  3) fan-in    selection of the best configuration and test evaluation

Usage: python -m src.pipeline
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import prepare, select_best
from .config import CONFIGS, N_WORKERS
from .utils import get_logger, set_seeds
from .worker import train_config

log = get_logger("pipeline")


def main():
    set_seeds()
    start = time.time()

    log.info("STEP 1/3: prepare data and features")
    prepare.main()

    log.info("STEP 2/3: fan-out, %d configurations on %d worker processes", len(CONFIGS), N_WORKERS)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(train_config, cfg["id"]): cfg["id"] for cfg in CONFIGS}
        for future in as_completed(futures):
            result = future.result()  # raises the error again if the worker failed
            log.info("worker for %s finished, val F1 %.4f", futures[future], result["val_f1"])

    log.info("STEP 3/3: fan-in, select the best configuration")
    select_best.main()

    log.info("pipeline finished in %.1f s", time.time() - start)


if __name__ == "__main__":
    main()
