"""Load and clean the SGCC electricity theft dataset.

The raw file (data/raw/sgcc/data.csv) has one row per customer:
    CONS_NO  - customer ID
    FLAG     - 1 = labelled as theft, 0 = normal
    one column per day (2014-01-01 to 2016-10-31) with daily usage in kWh

Cleaning steps, in order (see notebooks/01_data_quality.ipynb for the
checks that led to each decision):
    1. Load, reading the daily values as float32 to halve memory.
    2. Sort the day columns into real date order and add any missing day.
    3. Remove customers with no readings at all.
    4. Save each customer's share of missing days as a column.
    5. Cap extreme high values per customer with the IQR rule
       (k = 3, calculated on each customer's non-zero days).
    6. Fill gaps: interpolate the middle, backward fill the start,
       forward fill the end.
    7. Split customers into training and test sets (stratified).

Every step uses only each customer's own readings, so no information
leaks from the test set into the training set.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Names of the two non-reading columns in the raw file.
ID_COL = "CONS_NO"
LABEL_COL = "FLAG"
# Column added during cleaning.
MISSING_SHARE_COL = "missing_share"


def load_sgcc(csv_path):
    """Read the raw SGCC file.

    Memory optimization: pandas stores numbers as float64 (8 bytes) by
    default. Daily kWh values do not need that precision, so we read them
    as float32 (4 bytes), which halves the memory used by the readings.

    Args:
        csv_path: path to data.csv.

    Returns:
        DataFrame with CONS_NO, FLAG and one float32 column per day.
    """
    # Read only the header first, to learn the column names.
    columns = pd.read_csv(csv_path, nrows=0).columns
    # Every column except the ID and the label is a daily reading.
    dtypes = {col: np.float32 for col in columns if col not in (ID_COL, LABEL_COL)}
    dtypes[LABEL_COL] = np.int8  # 0 or 1 fits in one byte
    return pd.read_csv(csv_path, dtype=dtypes)


def split_info_and_readings(df):
    """Separate the customer info (ID, label) from the daily readings.

    Returns:
        (info, readings): info has CONS_NO and FLAG; readings has only
        the day columns. Both keep the same row index.
    """
    info = df[[ID_COL, LABEL_COL]]
    readings = df.drop(columns=[ID_COL, LABEL_COL])
    return info, readings


def sort_and_complete_dates(readings):
    """Put the day columns in date order and add any missing days.

    The raw columns are sorted as text ("2014/1/1", "2014/1/10", ...), so
    day 10 comes before day 2. We convert them to real dates and sort them.
    If a day is missing from the file, we add it as an empty column so the
    time series has no hole; it is filled later with the other gaps.

    Returns:
        (readings, added_days): readings with columns named "YYYY-MM-DD"
        in date order, and a list of the days that were added.
    """
    dates = pd.to_datetime(readings.columns, format="%Y/%m/%d")
    readings = readings.copy()
    readings.columns = dates
    readings = readings.sort_index(axis=1)

    # Every calendar day from the first to the last date.
    all_days = pd.date_range(dates.min(), dates.max(), freq="D")
    added_days = all_days.difference(dates)
    # reindex adds the missing days as empty (NaN) columns, keeping float32.
    readings = readings.reindex(columns=all_days).astype(np.float32)

    readings.columns = readings.columns.strftime("%Y-%m-%d")
    return readings, [d.strftime("%Y-%m-%d") for d in added_days]


def find_empty_customers(readings):
    """Return a True/False mask of customers with no readings at all."""
    return readings.isnull().all(axis=1)


def missing_share(readings):
    """Share of days with no reading, per customer (0.0 to 1.0)."""
    return readings.isnull().mean(axis=1).astype(np.float32)


def cap_outliers_iqr(readings, k=3.0):
    """Cap extreme high readings per customer using the IQR rule.

    For each customer, using only their own non-zero readings:
        Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
        upper limit = Q3 + k * IQR
    Readings above the limit are replaced by the limit (not deleted).

    Why k = 3 (Tukey's "extreme outlier" fence) instead of the usual 1.5:
    daily electricity usage is skewed (most days are low, some are
    naturally much higher). On the real SGCC data, k = 1.5 capped about 4%
    of all readings, mostly normal high-usage days. k = 3 capped about
    1.2% and still caught the isolated recording errors (e.g. 800,003 kWh
    for a customer who normally uses about 4 kWh).

    Why non-zero days only: for a customer with many zero days, Q1 and Q3
    are pulled down to almost 0, so the limit becomes tiny and their real
    usage would be capped. Using non-zero days measures their usual level
    on days they actually used electricity. Zeros themselves are never
    changed.

    Only high values are capped: low values and zeros can be real usage
    or a sign of theft, so they are left as they are.

    Customers whose IQR is 0 (e.g. only one distinct non-zero value) are
    skipped, because their limit would equal that value and nothing
    meaningful could be measured.

    Missing values (NaN) are ignored and stay missing.

    Args:
        readings: DataFrame of daily readings, one row per customer.
        k: IQR multiplier; 3 marks "extreme" outliers.

    Returns:
        (capped, stats): the capped readings and a dict with the number
        of readings capped, customers affected and customers skipped.
    """
    # Zeros (and missing days) become NaN here, so quantile() ignores them.
    nonzero = readings.where(readings > 0)
    q1 = nonzero.quantile(0.25, axis=1)
    q3 = nonzero.quantile(0.75, axis=1)
    iqr = q3 - q1
    upper = q3 + k * iqr

    # Customers with IQR = 0 get no limit (infinity), so nothing is capped.
    skipped = iqr == 0
    upper = upper.where(~skipped, np.inf)

    above = readings.gt(upper, axis=0)
    capped = readings.clip(upper=upper, axis=0).astype(np.float32)

    stats = {
        "readings_capped": int(above.to_numpy().sum()),
        "customers_capped": int(above.any(axis=1).sum()),
        "customers_skipped_iqr_zero": int(skipped.sum()),
    }
    return capped, stats


def fill_missing(readings):
    """Fill missing daily readings for each customer.

    1. Gaps with real readings on both sides: linear interpolation
       (a straight line between the neighbouring days). Usage changes
       gradually, and unlike forward fill this does not create artificial
       jumps that could look like sudden drops in usage.
    2. Gaps at the start of a customer's series: backward fill (copy the
       first real reading back).
    3. Gaps at the end: forward fill (copy the last real reading forward).

    Each customer is filled using only their own readings.
    """
    filled = readings.interpolate(method="linear", axis=1, limit_area="inside")
    filled = filled.bfill(axis=1).ffill(axis=1)
    return filled.astype(np.float32)


def clean_sgcc(raw):
    """Run all cleaning steps on the raw SGCC data.

    Args:
        raw: DataFrame from load_sgcc().

    Returns:
        (clean, report): the cleaned DataFrame (CONS_NO, FLAG,
        missing_share, then one column per day) and a dict of numbers
        describing what each step did, for the data-quality summary.
    """
    report = {"customers_raw": len(raw), "days_raw": raw.shape[1] - 2}
    info, readings = split_info_and_readings(raw)
    report["missing_share_raw"] = float(readings.isnull().to_numpy().mean())

    # Step 2: date order and missing days.
    readings, added_days = sort_and_complete_dates(readings)
    report["days_added"] = added_days

    # Step 3: drop customers with no readings at all.
    empty = find_empty_customers(readings)
    report["customers_removed_empty"] = int(empty.sum())
    readings, info = readings[~empty], info[~empty]

    # Step 4: record the missing share before any filling. Added days are
    # not counted, since they are missing for everyone, not per customer.
    original_days = readings.drop(columns=added_days)
    share = missing_share(original_days)

    # Step 5: cap outliers on real readings, before filling, so extreme
    # values cannot spread into the filled-in days.
    readings, cap_stats = cap_outliers_iqr(readings)
    report.update(cap_stats)

    # Step 6: fill all gaps.
    readings = fill_missing(readings)
    report["missing_after_cleaning"] = int(readings.isnull().to_numpy().sum())

    clean = pd.concat(
        [info.reset_index(drop=True),
         share.rename(MISSING_SHARE_COL).reset_index(drop=True),
         readings.reset_index(drop=True)],
        axis=1,
    )
    report["customers_clean"] = len(clean)
    report["days_clean"] = readings.shape[1]
    return clean, report


def split_sgcc(clean, test_size, seed):
    """Split customers into training and test sets.

    Each row is one customer, so no customer can be in both sets.
    Stratifying by FLAG keeps the same theft share (about 8.5%) in both.

    Args:
        clean: DataFrame from clean_sgcc().
        test_size: share of customers for the test set, e.g. 0.2.
        seed: random seed from config.yaml, so the split always repeats.

    Returns:
        (train, test) DataFrames.
    """
    train, test = train_test_split(
        clean,
        test_size=test_size,
        stratify=clean[LABEL_COL],
        random_state=seed,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
