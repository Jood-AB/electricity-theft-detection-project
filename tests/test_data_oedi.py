"""Tests for the OEDI cleaning steps, using small made-up data.

Run with:  python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src import data_oedi


@pytest.fixture
def short_year(monkeypatch):
    """Pretend a year has 3 hours, so test tables stay tiny."""
    monkeypatch.setattr(data_oedi, "HOURS_PER_YEAR", 3)


def make_raw(types=("Hospital", "School"), recordings_per_type=4, hours=3):
    """A tiny raw OEDI-style table with the same layout as the real file.

    Recordings follow the real order: all types once, then repeated.
    One broken row (building type "0") is added at the end.
    """
    rows = []
    for _ in range(recordings_per_type):
        for building_type in types:
            for hour in range(hours):
                label = "Theft1" if hour == 0 else "Normal"
                rows.append((building_type, label))
    rows.append(("0", "Theft2"))  # broken row

    df = pd.DataFrame(rows, columns=["Class", "theft"])
    df.insert(0, "0", range(len(df)))  # row-number column
    for raw_name in data_oedi.RENAME:
        if raw_name != "Class":
            df[raw_name] = np.float32(1.5)
    return df


def test_clean_drops_broken_rows_and_row_number(short_year):
    clean, report = data_oedi.clean_oedi(make_raw())

    assert report["rows_removed_broken"] == 1
    assert "0" not in clean.columns
    assert (clean["building_type"] != "0").all()
    # 2 types x 4 recordings = 8 recordings, each with hours 0..2.
    assert report["recordings"] == 8
    assert clean.groupby("recording_id")["hour"].apply(list).tolist() == [[0, 1, 2]] * 8


def test_labels_binary_and_type_kept(short_year):
    clean, _ = data_oedi.clean_oedi(make_raw())

    assert set(clean["theft"].unique()) == {0, 1}
    assert (clean.loc[clean["theft_type"] == "Normal", "theft"] == 0).all()
    assert (clean.loc[clean["theft_type"] == "Theft1", "theft"] == 1).all()
    assert "elec_facility_kw" in clean.columns
    assert clean["elec_facility_kw"].dtype == np.float32


def test_wrong_recording_length_raises():
    # With the real 8,760-hour year, 3-hour recordings must be rejected.
    with pytest.raises(ValueError):
        data_oedi.clean_oedi(make_raw())


def test_split_one_recording_per_type(short_year):
    clean, _ = data_oedi.clean_oedi(make_raw())
    train, test = data_oedi.split_oedi(clean, test_recordings_per_type=1, seed=42)

    # No recording appears in both sets.
    assert set(train["recording_id"]).isdisjoint(test["recording_id"])
    # Exactly one test recording per building type; every type is in training.
    assert test.groupby("building_type")["recording_id"].nunique().tolist() == [1, 1]
    assert set(train["building_type"]) == {"Hospital", "School"}
    assert len(test) + len(train) == len(clean)

    # Same seed -> same test recordings.
    _, test2 = data_oedi.split_oedi(clean, test_recordings_per_type=1, seed=42)
    assert sorted(test["recording_id"].unique()) == sorted(test2["recording_id"].unique())
