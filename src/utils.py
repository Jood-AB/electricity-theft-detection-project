"""Small helper functions shared by the whole project."""

import os
import random

import numpy as np


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
    # Python's hashing, which can affect the order of some operations.
    os.environ["PYTHONHASHSEED"] = str(seed)
