"""Basic checks that the project setup works.

Run all tests from the project folder with:  python -m pytest
"""

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.utils import set_seed


def test_config_loads_expected_settings():
    """config.yaml can be read and has the settings everyone relies on."""
    config = load_config()

    assert isinstance(config["random_seed"], int)
    # The test share must be a fraction between 0 and 1.
    assert 0 < config["test_size"] < 1

    # Both datasets must have separate raw and processed folders.
    for name in ["sgcc_raw", "oedi_raw", "sgcc_processed", "oedi_processed", "outputs"]:
        assert name in config["paths"]


def test_config_paths_are_inside_project():
    """All folder paths point inside the project folder."""
    config = load_config()
    for path in config["paths"].values():
        assert PROJECT_ROOT in path.parents


def test_set_seed_makes_randomness_repeatable():
    """The same seed gives exactly the same random numbers."""
    set_seed(42)
    first_run = np.random.rand(5)

    set_seed(42)
    second_run = np.random.rand(5)

    assert np.array_equal(first_run, second_run)
