"""Tests for the feature functions, using small made-up data.

Run with:  python -m pytest
"""

import numpy as np
import pandas as pd

from src import features


def make_sgcc():
    """A tiny cleaned SGCC-style table: 3 customers, 10 days."""
    days = [f"2014-01-{d:02d}" for d in range(1, 11)]
    readings = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],           # a: steady
        [10, 10, 10, 10, 10, 10, 10, 1, 10, 10],  # b: one sudden drop on day 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # c: always zero
    ], dtype=np.float32)
    df = pd.DataFrame(readings, columns=days)
    df.insert(0, "missing_share", np.array([0.0, 0.1, 0.5], dtype=np.float32))
    df.insert(0, "FLAG", [0, 1, 0])
    df.insert(0, "CONS_NO", ["a", "b", "c"])
    return df


def make_oedi(hours=30):
    """Two recordings in the processed OEDI layout.

    Recording 1 always uses 100 kW and recording 2 always uses 1 kW, so any
    window that mixes the two recordings gives a clearly wrong value.
    """
    parts = []
    for recording_id, kw in [(1, 100.0), (2, 1.0)]:
        parts.append(pd.DataFrame({
            "recording_id": recording_id,
            "building_type": "Hospital",
            "hour": range(hours),
            "elec_facility_kw": kw,
            "theft": 0,
            "theft_type": "Normal",
        }))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- SGCC

def test_sgcc_basic_statistics():
    feats = features.sgcc_features(make_sgcc())
    b = make_sgcc().iloc[1, 3:].to_numpy(dtype=float)

    assert np.isclose(feats.loc[1, "mean_use"], 9.1)
    assert np.isclose(feats.loc[1, "std_use"], np.std(b, ddof=1), rtol=1e-5)
    assert feats.loc[1, "min_use"] == 1
    assert feats.loc[1, "max_use"] == 10
    assert feats["zero_days"].tolist() == [0, 0, 10]


def test_sgcc_one_sudden_drop_is_found():
    feats = features.sgcc_features(make_sgcc())
    # Only customer b drops (10 -> 1); steady and all-zero customers do not.
    assert feats["sudden_drops"].tolist() == [0, 1, 0]


def test_long_low_period_counts_as_one_drop():
    readings = pd.DataFrame([[10, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1]], dtype=np.float32)
    assert features.count_sudden_drops(readings).tolist() == [1]


def test_slow_decline_is_not_a_sudden_drop():
    readings = pd.DataFrame([[10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6]], dtype=np.float32)
    assert features.count_sudden_drops(readings).tolist() == [0]


def test_relative_std_is_zero_for_always_zero_customer():
    feats = features.sgcc_features(make_sgcc())
    # Customer c always uses 0: std / mean would be 0 / 0, so it must be 0.
    assert feats.loc[2, "relative_std"] == 0
    assert not feats.isnull().any().any()


def test_sgcc_features_exclude_id_and_label():
    df = make_sgcc()
    feats = features.sgcc_features(df)

    assert "CONS_NO" not in feats.columns
    assert "FLAG" not in feats.columns
    # One row per customer, in the same order as the input.
    assert feats.index.equals(df.index)
    # missing_share is passed through unchanged.
    assert feats["missing_share"].tolist() == df["missing_share"].tolist()


# ---------------------------------------------------------------- OEDI

def test_oedi_hour_of_day():
    feats = features.oedi_features(make_oedi(hours=30))
    first_recording = feats["hour_of_day"].iloc[:30].tolist()
    assert first_recording == list(range(24)) + list(range(6))


def test_oedi_window_stays_inside_one_recording():
    df = make_oedi()
    feats = features.oedi_features(df)

    first_hour_of_rec2 = df.index[(df["recording_id"] == 2) & (df["hour"] == 0)][0]
    # Recording 2 always uses 1 kW. If the window crossed into recording 1
    # (100 kW), the window average would be far above 1, so this hour would
    # look like a drop and the spread would not be 0.
    assert feats.loc[first_hour_of_rec2, "elec_vs_mean_24h"] == 1.0
    assert feats.loc[first_hour_of_rec2, "elec_rel_std_24h"] == 0.0


def test_oedi_shuffled_rows_give_the_same_features():
    df = make_oedi()
    df.loc[df["recording_id"] == 1, "elec_facility_kw"] = np.arange(30, dtype=float)
    expected = features.oedi_features(df)

    shuffled = df.sample(frac=1, random_state=42)
    got = features.oedi_features(shuffled)

    # Rows come back in the order they were given ...
    assert got.index.equals(shuffled.index)
    # ... and each row has the same values as with sorted input.
    pd.testing.assert_frame_equal(got.sort_index(), expected)


def test_oedi_drop_to_zero_and_no_division_by_zero():
    df = make_oedi(hours=50)
    rec1 = df["recording_id"] == 1
    # 24 hours at 10 kW, then 26 hours at 0 kW.
    df.loc[rec1, "elec_facility_kw"] = [10.0] * 24 + [0.0] * 26
    feats = features.oedi_features(df)[rec1.to_numpy()]

    assert feats["elec_vs_mean_24h"].iloc[23] == 1.0   # usual level
    assert feats["elec_rel_std_24h"].iloc[23] == 0.0   # steady use
    assert feats["elec_vs_mean_24h"].iloc[24] == 0.0   # sudden drop to zero
    # Hour 49: the whole window is 0, so 0 / 0 must give 0, not NaN.
    assert feats["elec_vs_mean_24h"].iloc[49] == 0.0
    assert feats["elec_rel_std_24h"].iloc[49] == 0.0
    assert not feats.isnull().any().any()


def test_oedi_relative_spread_ignores_building_size():
    df = make_oedi(hours=48)
    # Same daily pattern in both recordings, but recording 1 is 100 times bigger.
    pattern = np.tile([1.0, 3.0], 24)
    df.loc[df["recording_id"] == 1, "elec_facility_kw"] = 100 * pattern
    df.loc[df["recording_id"] == 2, "elec_facility_kw"] = pattern
    feats = features.oedi_features(df)

    big = feats[(df["recording_id"] == 1).to_numpy()]["elec_rel_std_24h"].to_numpy()
    small = feats[(df["recording_id"] == 2).to_numpy()]["elec_rel_std_24h"].to_numpy()
    np.testing.assert_allclose(big, small, rtol=1e-5)
    assert small[-1] > 0   # the pattern does jump around


def test_oedi_features_have_no_size_based_columns():
    # Average and spread in kW mostly say how big the building is, not
    # whether something unusual happens, so only relative features are kept.
    feats = features.oedi_features(make_oedi())
    assert list(feats.columns) == ["hour_of_day", "elec_vs_mean_24h", "elec_rel_std_24h"]


def test_oedi_features_exclude_ids_and_labels():
    df = make_oedi()
    feats = features.oedi_features(df)

    for column in ["recording_id", "building_type", "theft", "theft_type"]:
        assert column not in feats.columns
    assert feats.index.equals(df.index)
