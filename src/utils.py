"""Small helper functions shared by the whole project."""

import os
import random

import numpy as np
import pandas as pd

# Rows read at a time by read_csv_in_chunks.
READ_CHUNK_ROWS = 5000


def set_seed(seed):
    """Fix all sources of randomness so results repeat exactly.

    Call this once at the start of every script, using the seed from
    config.yaml:

        config = load_config()
        set_seed(config["random_seed"])

    Note: scikit-learn functions (train_test_split, RandomForestClassifier,
    cross-validation) also take their own random_state argument. Always pass
    the same seed there too, e.g. random_state=config["random_seed"].

    Args:
        seed: any whole number. The same number always gives the same
            "random" choices.
    """
    # Python's built-in random module.
    random.seed(seed)
    # NumPy, which pandas and scikit-learn use under the hood.
    np.random.seed(seed)
    # Python's hashing, which can affect the order of some operations. This
    # only reaches processes started after this line (for example parallel
    # workers); the running Python process keeps its own hash seed.
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_csv_in_chunks(csv_path, dtype, chunk_rows=READ_CHUNK_ROWS):
    """Read a CSV file a few thousand rows at a time and join the pieces.

    Memory optimization (PR 7): passing dtype=float32 to pd.read_csv halves
    the memory of the finished table, but on the whole SGCC file the reader
    still needs far more memory while it works (about 1.3 GB at its peak
    for the 144 MB training table). Reading `chunk_rows` rows at a time
    keeps only one small piece in the reader at once, which cut the peak
    to about 0.3 GB. The result is exactly the same table. Numbers are in
    outputs/results/efficiency.csv.

    Args:
        csv_path: path to the CSV file.
        dtype: column types, as for pd.read_csv (e.g. {"col": np.float32}).
        chunk_rows: number of rows read at a time.

    Returns:
        DataFrame with the whole file.
    """
    chunks = pd.read_csv(csv_path, dtype=dtype, chunksize=chunk_rows)
    return pd.concat(chunks, ignore_index=True)
