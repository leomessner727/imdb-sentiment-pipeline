"""Step 3 of the pipeline (fan-in): collects the results of all workers,
selects the configuration with the best validation F1 and evaluates it
once on the test set. The final metrics are saved to data/final_metrics.json.
"""
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import CONFIGS, FINAL_METRICS_FILE, MODELS_DIR, RESULTS_DIR
from .prepare import build_features
from .utils import get_logger, read_json, write_json_atomic

log = get_logger("select_best")


def collect_results():
    """Reads the result file of every configuration; fails if one is missing."""
    results, missing = [], []
    for cfg in CONFIGS:
        path = RESULTS_DIR / f"{cfg['id']}.json"
        if path.exists():
            results.append(read_json(path))
        else:
            missing.append(cfg["id"])
    if missing:
        raise RuntimeError(f"missing worker results for {missing}: run the workers first")
    return results


def print_results_table(results):
    """Prints one line per configuration (including the process that trained it)."""
    print()
    print(f"{'config':<8}{'tfidf':<7}{'C':>5}{'val_acc':>10}{'val_f1':>10}{'train_s':>9}{'pid':>8}")
    for r in sorted(results, key=lambda r: r["config_id"]):
        print(f"{r['config_id']:<8}{r['tfidf']:<7}{r['C']:>5.1f}{r['val_accuracy']:>10.4f}"
              f"{r['val_f1']:>10.4f}{r['train_seconds']:>9.1f}{r['pid']:>8}")
    print()


def main():
    results = collect_results()
    print_results_table(results)

    best = max(results, key=lambda r: r["val_f1"])
    log.info("best configuration: %s (val F1 %.4f)", best["config_id"], best["val_f1"])

    # Final evaluation of the best model on the test set
    data = build_features(best["tfidf"])
    model = joblib.load(MODELS_DIR / f"{best['config_id']}.joblib")
    y_pred = model.predict(data["X_test"])

    final = {
        "best_config": best["config_id"],
        "tfidf": best["tfidf"],
        "C": best["C"],
        "val_f1": best["val_f1"],
        "test_accuracy": float(accuracy_score(data["y_test"], y_pred)),
        "test_f1": float(f1_score(data["y_test"], y_pred)),
        "all_configs": results,
    }
    write_json_atomic(FINAL_METRICS_FILE, final)

    print(classification_report(data["y_test"], y_pred,
                                target_names=["negative", "positive"], digits=4))
    log.info("TEST accuracy %.4f, TEST F1 %.4f, saved to %s",
             final["test_accuracy"], final["test_f1"], FINAL_METRICS_FILE)


if __name__ == "__main__":
    main()
