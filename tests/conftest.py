"""Shared test data: tiny SGCC- and OEDI-shaped files with every column the
feature code needs, so tuning and error analysis can run end to end."""

import numpy as np
import pandas as pd
import pytest

from src import models


def make_sgcc(n, rng):
    """n customers x 20 days. Theft customers' use drops in the second half."""
    flag = (rng.random(n) < 0.25).astype(int)
    readings = rng.uniform(5, 10, size=(n, 20))
    readings[flag == 1, 10:] *= rng.uniform(0.1, 0.6, size=(int(flag.sum()), 1))
    df = pd.DataFrame(readings.round(3), columns=[f"2014-01-{d:02d}" for d in range(1, 21)])
    df.insert(0, "missing_share", rng.uniform(0, 0.3, n).round(3))
    df.insert(0, "FLAG", flag)
    df.insert(0, "CONS_NO", [f"C{i}" for i in range(n)])
    return df


def make_oedi(recordings, rng, hours=48):
    """`hours` hours per recording; theft hours use less electricity."""
    n = hours * len(recordings)
    theft = (rng.random(n) < 0.4).astype(int)
    return pd.DataFrame({
        "recording_id": np.repeat(recordings, hours),
        "building_type": np.repeat([f"type{r % 2}" for r in recordings], hours),
        "hour": np.tile(range(hours), len(recordings)),
        "elec_facility_kw": rng.uniform(5, 10, n) * np.where(theft == 1, 0.4, 1.0),
        "theft": theft,
        "theft_type": np.where(theft == 1, np.where(rng.random(n) < 0.5, "Theft1", "Theft2"),
                               "Normal"),
    })


@pytest.fixture
def featured_project(tmp_path, monkeypatch):
    """Write tiny train/test files and point all output folders at tmp_path."""
    rng = np.random.default_rng(0)
    files = {"sgcc": (make_sgcc(150, rng), make_sgcc(50, rng)),
             "oedi": (make_oedi([1, 2, 3, 4, 5, 6], rng), make_oedi([7, 8], rng))}
    for name, (train, test) in files.items():
        (tmp_path / "data" / name).mkdir(parents=True)
        train.to_csv(tmp_path / "data" / name / "train.csv", index=False)
        test.to_csv(tmp_path / "data" / name / "test.csv", index=False)

    monkeypatch.setattr(models, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(models, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(models, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    return tmp_path
