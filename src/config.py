"""Load the shared project settings from config.yaml.

All scripts should get their settings (random seed, test size, folder
paths) from load_config() instead of typing values directly. That way
every member uses exactly the same settings.
"""

from pathlib import Path

import yaml

# The project's main folder: one level above this file (src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default location of the settings file.
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config(config_file=CONFIG_FILE):
    """Read config.yaml and return its settings as a dictionary.

    Folder paths in the file are written relative to the project folder.
    Here we turn them into full paths, so the code works no matter which
    folder you run it from.

    Args:
        config_file: path to the YAML settings file. Defaults to the
            project's config.yaml.

    Returns:
        dict with keys such as "random_seed", "test_size" and "paths".
        Each value in "paths" is a pathlib.Path.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert every relative path into a full path inside the project.
    config["paths"] = {
        name: PROJECT_ROOT / relative_path
        for name, relative_path in config["paths"].items()
    }
    return config
