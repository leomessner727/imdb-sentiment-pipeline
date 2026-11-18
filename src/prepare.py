"""Step 1 of the pipeline: data ingestion and feature extraction.

- "split":    downloads IMDB from Hugging Face and splits the training reviews
              into training and validation set (stratified, fixed seed)
- "features": computes the TF-IDF matrices for one feature variant

Both steps are memoized (see memo.py): they are computed only once.
"""
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from .config import CONFIGS, DATASET_NAME, SEED, TFIDF_VARIANTS, VAL_SIZE
from .memo import memoized_step
from .utils import get_logger, set_seeds

log = get_logger("prepare")


def clean_text(text):
    """Removes the HTML line breaks contained in many IMDB reviews."""
    return text.replace("<br />", " ")


def split_params():
    """Parameters that define the split (part of the memoization key)."""
    return {"dataset": DATASET_NAME, "val_size": VAL_SIZE, "seed": SEED}


def load_split():
    """Memoized step "split": texts and labels for train, validation and test."""

    def compute():
        ds = load_dataset(DATASET_NAME)
        texts = [clean_text(t) for t in ds["train"]["text"]]
        labels = list(ds["train"]["label"])
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=VAL_SIZE, random_state=SEED, stratify=labels
        )
        test_texts = [clean_text(t) for t in ds["test"]["text"]]
        test_labels = list(ds["test"]["label"])
        log.info("split: %d train, %d validation, %d test reviews",
                 len(train_texts), len(val_texts), len(test_texts))
        return {
            "train": (train_texts, np.array(train_labels)),
            "val": (val_texts, np.array(val_labels)),
            "test": (test_texts, np.array(test_labels)),
        }

    return memoized_step("split", split_params(), compute)


def build_features(variant):
    """Memoized step "features": TF-IDF matrices for one variant ("uni" or "bi").

    The vectorizer is fitted on the training texts only, so no information
    from validation or test data leaks into the features.
    """
    tfidf_params = TFIDF_VARIANTS[variant]
    params = {"split": split_params(), "variant": variant, "tfidf": tfidf_params}

    def compute():
        split = load_split()
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(tfidf_params["ngram_range"]),
            min_df=tfidf_params["min_df"],
            max_features=tfidf_params["max_features"],
            sublinear_tf=tfidf_params["sublinear_tf"],
        )
        X_train = vectorizer.fit_transform(split["train"][0])
        X_val = vectorizer.transform(split["val"][0])
        X_test = vectorizer.transform(split["test"][0])
        # The list of discarded terms is only kept for inspection and can be
        # very large with bigrams: drop it before saving.
        if "stop_words_" in vars(vectorizer):
            del vectorizer.stop_words_
        log.info("features '%s': %d terms", variant, X_train.shape[1])
        return {
            "vectorizer": vectorizer,
            "X_train": X_train, "y_train": split["train"][1],
            "X_val": X_val, "y_val": split["val"][1],
            "X_test": X_test, "y_test": split["test"][1],
        }

    return memoized_step("features", params, compute)


def main():
    """Prepares the split and every feature variant used by the configurations."""
    set_seeds()
    load_split()
    for variant in sorted({cfg["tfidf"] for cfg in CONFIGS}):
        build_features(variant)
    log.info("data preparation done")


if __name__ == "__main__":
    main()
