"""Tests for the SGCC cleaning steps, using small made-up data.

Run with:  python -m pytest
"""

import numpy as np
import pandas as pd

from src import data_sgcc


def make_raw():
    """A tiny raw SGCC-style table: 4 customers, 5 days, dates out of order.

    The days 2014/1/4 and 2014/1/6 to 2014/1/9 are deliberately missing
    from the columns.
    """
    return pd.DataFrame({
        "CONS_NO": ["a", "b", "c", "d"],
        "FLAG": [0, 1, 0, 1],
        # Text-sorted order, like the real file: 1, 10, 2, 3, 5 (4 missing).
        "2014/1/1": [1.0, np.nan, np.nan, 2.0],
        "2014/1/10": [1.0, 5.0, np.nan, 2.0],
        "2014/1/2": [1.0, np.nan, np.nan, 2.0],
        "2014/1/3": [1.0, 5.0, np.nan, 2.0],
        "2014/1/5": [1.0, 5.0, np.nan, 2.0],
    })


def test_dates_sorted_and_missing_day_added():
    _, readings = data_sgcc.split_info_and_readings(make_raw())
    sorted_readings, added = data_sgcc.sort_and_complete_dates(readings)

    assert list(sorted_readings.columns)[:5] == [
        "2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"]
    assert sorted_readings.columns[-1] == "2014-01-10"
    # Jan 4 and Jan 6-9 were not in the file, so they are added.
    assert "2014-01-04" in added
    assert sorted_readings["2014-01-04"].isnull().all()


def test_empty_customers_found():
    _, readings = data_sgcc.split_info_and_readings(make_raw())
    empty = data_sgcc.find_empty_customers(readings)
    assert empty.tolist() == [False, False, True, False]


def test_iqr_caps_only_the_spike():
    readings = pd.DataFrame([[5, 6, 4, 7, 5, 800003, 6, 5]], dtype=np.float32)
    capped, stats = data_sgcc.cap_outliers_iqr(readings)

    values = capped.iloc[0].tolist()
    # The spike is brought down to the limit, far below 800003 ...
    assert values[5] < 20
    # ... and every normal reading is unchanged.
    assert values[:5] == [5, 6, 4, 7, 5]
    assert values[6:] == [6, 5]
    assert stats["readings_capped"] == 1


def test_iqr_protects_customers_with_many_zero_days():
    # 75% zeros. On all days the limit would be about 3, capping real usage.
    # On non-zero days ([3, 4]) the limit is above 4, so nothing changes.
    readings = pd.DataFrame([[0, 0, 0, 0, 0, 0, 3, 4]], dtype=np.float32)
    capped, stats = data_sgcc.cap_outliers_iqr(readings)

    assert capped.iloc[0].tolist() == [0, 0, 0, 0, 0, 0, 3, 4]
    assert stats["readings_capped"] == 0


def test_iqr_skips_customers_with_zero_iqr():
    # Only one non-zero value: Q1 = Q3, so IQR = 0. Must be skipped.
    readings = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 4]], dtype=np.float32)
    capped, stats = data_sgcc.cap_outliers_iqr(readings)

    assert capped.iloc[0].tolist() == [0, 0, 0, 0, 0, 0, 0, 4]
    assert stats["customers_skipped_iqr_zero"] == 1


def test_iqr_leaves_missing_values_missing():
    readings = pd.DataFrame([[5, np.nan, 4, 7, 5, 6]], dtype=np.float32)
    capped, _ = data_sgcc.cap_outliers_iqr(readings)
    assert np.isnan(capped.iloc[0, 1])


def test_fill_interpolates_middle_and_fills_edges():
    readings = pd.DataFrame([[np.nan, 10, np.nan, np.nan, np.nan, 18, np.nan]],
                            dtype=np.float32)
    filled = data_sgcc.fill_missing(readings).iloc[0].tolist()

    # Start: backward fill. Middle: straight line 10 -> 18. End: forward fill.
    assert filled == [10, 10, 12, 14, 16, 18, 18]


def test_clean_sgcc_end_to_end():
    clean, report = data_sgcc.clean_sgcc(make_raw())

    assert report["customers_removed_empty"] == 1
    assert len(clean) == 3
    assert clean.isnull().sum().sum() == 0
    # Missing share is measured on the original days only (5 days here).
    shares = dict(zip(clean["CONS_NO"], clean["missing_share"]))
    assert shares["a"] == 0.0
    assert np.isclose(shares["b"], 2 / 5)
    # Readings are stored as float32 to save memory.
    assert clean["2014-01-01"].dtype == np.float32


def test_split_has_no_shared_customers_and_keeps_theft_share():
    n = 100
    clean = pd.DataFrame({
        "CONS_NO": [f"c{i}" for i in range(n)],
        "FLAG": [1 if i < 10 else 0 for i in range(n)],  # 10% theft
        "2014-01-01": np.ones(n, dtype=np.float32),
    })
    train, test = data_sgcc.split_sgcc(clean, test_size=0.2, seed=42)

    assert len(train) == 80 and len(test) == 20
    assert set(train["CONS_NO"]).isdisjoint(test["CONS_NO"])
    # Stratified: both sets keep the 10% theft share.
    assert train["FLAG"].mean() == 0.1
    assert test["FLAG"].mean() == 0.1

    # Same seed -> same split.
    train2, _ = data_sgcc.split_sgcc(clean, test_size=0.2, seed=42)
    assert train["CONS_NO"].tolist() == train2["CONS_NO"].tolist()
