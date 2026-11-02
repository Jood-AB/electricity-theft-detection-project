"""Build the features (the numbers the models learn from) for both datasets.

SGCC and OEDI each have their own function; the two are never mixed.

    sgcc_features(df)  - one row per customer: summary numbers of the
                         customer's daily readings
    oedi_features(df)  - one row per hour: hour of day and statistics of
                         the last 24 hours inside the same recording

Why these features (see notebooks/02_eda.ipynb):
    - SGCC theft customers do not use less electricity; their use is more
      irregular. So we measure variability and sudden drops, not only the
      average.
    - Usage has normal rhythms (seasons, day and night), so each customer
      or building is compared with its own earlier use, not with a fixed
      limit.
    - OEDI theft comes in stretches of about 24 hours, and two theft types
      keep a normal amount of use but change the pattern. So we add the
      hour of day and 24-hour window statistics.

No leakage:
    - Every SGCC feature uses only that customer's own row.
    - Every OEDI window uses only the current hour and earlier hours of the
      same recording
      (grouped by recording_id and sorted by hour first).
    - Nothing is learned from the data (no fitted scaler or threshold), so
      the same functions can be applied to the training and test sets.
      If a model needs scaling, fit the scaler on the training set only.
    - IDs (CONS_NO, recording_id) and labels (FLAG, theft, theft_type) are
      never returned as features.
"""

import numpy as np
import pandas as pd

from src.data_sgcc import ID_COL, LABEL_COL, MISSING_SHARE_COL

# SGCC: a day is a sudden drop when usage falls below half of the
# customer's average over the previous 7 days.
DROP_WINDOW_DAYS = 7
DROP_RATIO = 0.5
# Customers handled at a time when counting drops, so the temporary arrays
# stay small (see count_sudden_drops).
DROP_BATCH_SIZE = 2000

# OEDI: theft stretches typically last about 24 hours.
OEDI_WINDOW_HOURS = 24
OEDI_USAGE_COL = "elec_facility_kw"
HOURS_PER_DAY = 24


# ---------------------------------------------------------------- SGCC

def sgcc_day_columns(df):
    """Return the names of the daily reading columns of a cleaned SGCC table."""
    return [col for col in df.columns if col not in (ID_COL, LABEL_COL, MISSING_SHARE_COL)]


def count_sudden_drops(readings, window=DROP_WINDOW_DAYS, ratio=DROP_RATIO,
                       batch_size=DROP_BATCH_SIZE):
    """Count how many times each customer's usage suddenly drops.

    A day is "low" when its usage is below `ratio` times the customer's
    average of the previous `window` days. A run of low days in a row
    counts as ONE drop, so a long low period is not counted many times.

    The first `window` days have no full history and are never low.
    A previous average of 0 is never a drop (there was nothing to drop).

    Efficiency (PR 7): the first version used pandas rolling() on a
    transposed copy of all readings. This version gives exactly the same
    counts but works on NumPy arrays, a batch of customers at a time, so it
    is about 8 times faster and its temporary arrays cover only
    `batch_size` customers instead of all of them (before/after numbers in
    outputs/results/efficiency.csv).

    Args:
        readings: DataFrame of daily readings, one row per customer, with no
            missing values (true after cleaning).
        window: number of previous days to average.
        ratio: how far usage must fall, e.g. 0.5 = below half.
        batch_size: number of customers handled at a time.

    Returns:
        Series with the number of drops per customer.
    """
    values = readings.to_numpy()
    counts = np.zeros(len(values), dtype=np.int64)
    for start in range(0, len(values), batch_size):
        # float64 for the running totals, so rounding cannot change a count.
        batch = values[start:start + batch_size].astype(np.float64)
        counts[start:start + batch_size] = _count_drops_in_batch(batch, window, ratio)
    return pd.Series(counts, index=readings.index)


def _count_drops_in_batch(batch, window, ratio):
    """count_sudden_drops for one NumPy array of customers (rows) x days."""
    n_days = batch.shape[1]
    # running[:, t] is the sum of the first t days, so the sum of the
    # `window` days before day t is running[:, t] - running[:, t - window].
    running = np.zeros((batch.shape[0], n_days + 1))
    np.cumsum(batch, axis=1, out=running[:, 1:])
    previous_avg = np.full(batch.shape, np.nan)
    previous_avg[:, window:] = (running[:, window:n_days] - running[:, :n_days - window]) / window

    # Comparisons with NaN (the first `window` days) are simply False.
    with np.errstate(invalid="ignore"):
        low = (batch < ratio * previous_avg) & (previous_avg > 0)

    # A drop starts on a low day whose previous day was not low.
    drop_starts = low.copy()
    drop_starts[:, 1:] &= ~low[:, :-1]
    return drop_starts.sum(axis=1)


def sgcc_features(df):
    """Build the SGCC features: one row per customer.

    Args:
        df: cleaned SGCC data (data/processed/sgcc/train.csv or test.csv).

    Returns:
        DataFrame with the same row index as df and the columns:
            mean_use       average daily use (kWh)
            std_use        standard deviation: how much use jumps around
            min_use        lowest daily use
            max_use        highest daily use
            zero_days      number of days with a reading of 0
            sudden_drops   number of sudden drops (see count_sudden_drops)
            relative_std   std_use / mean_use, so large and small users
                           can be compared fairly
            missing_share  share of days that were missing in the raw data
    """
    readings = df[sgcc_day_columns(df)]
    mean_use = readings.mean(axis=1)
    std_use = readings.std(axis=1)

    features = pd.DataFrame({
        "mean_use": mean_use,
        "std_use": std_use,
        "min_use": readings.min(axis=1),
        "max_use": readings.max(axis=1),
        "zero_days": (readings == 0).sum(axis=1),
        "sudden_drops": count_sudden_drops(readings),
        # A customer who always uses 0 would give 0 / 0, so we use 0 instead.
        "relative_std": (std_use / mean_use.replace(0, np.nan)).fillna(0),
        "missing_share": df[MISSING_SHARE_COL],
    }, index=df.index)

    # float32 halves the memory, like the cleaned readings.
    return features.astype(np.float32)


# ---------------------------------------------------------------- OEDI

def oedi_features(df, window=OEDI_WINDOW_HOURS, usage_col=OEDI_USAGE_COL):
    """Build the OEDI features: one row per hour.

    The rows are grouped by recording_id and sorted by hour before any
    window is calculated, so a window never mixes two buildings and always
    looks at hours in time order. Each window covers the current hour and
    the hours just before it; the first hours of a recording use the hours
    available so far.

    Args:
        df: processed OEDI data (data/processed/oedi/train.csv or test.csv).
        window: number of hours in each window.
        usage_col: the usage column to summarise (total electricity).

    Returns:
        DataFrame with the same row index and order as df and the columns:
            hour_of_day       0 (midnight) to 23
            elec_vs_mean_24h  this hour's use divided by the window average
                              (1 = usual, near 0 = sudden drop)
            elec_rel_std_24h  window standard deviation divided by the window
                              average (close to 0 when use is unnaturally flat)

        Both window features are ratios, so a big and a small building with
        the same pattern get the same values. The window average and spread
        in kW are not returned: on the training data, the building alone
        explains 70-79% of their variation, so they mostly say how big the
        building is (the same idea as relative_std in SGCC).
    """
    # Sort by recording, then by hour, so every window sees hours in order.
    ordered = df.sort_values(["recording_id", "hour"])
    usage = ordered[usage_col]
    by_recording = usage.groupby(ordered["recording_id"])

    mean_24h = by_recording.transform(lambda s: s.rolling(window, min_periods=1).mean())
    # A window of one hour has no standard deviation (NaN), so use 0.
    std_24h = by_recording.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0)

    # A window that is all 0 would give 0 / 0, so we use 0 instead.
    safe_mean = mean_24h.replace(0, np.nan)
    features = pd.DataFrame({
        "hour_of_day": ordered["hour"] % HOURS_PER_DAY,
        "elec_vs_mean_24h": (usage / safe_mean).fillna(0),
        "elec_rel_std_24h": (std_24h / safe_mean).fillna(0),
    })

    # Put the rows back in the order they were given.
    return features.loc[df.index].astype(np.float32)
