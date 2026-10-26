"""Load and clean the OEDI-based theft detection dataset.

The raw file (data/raw/oedi/df.csv) has one row per HOUR for one building:
    0        - row number (not useful, dropped)
    10 columns of hourly electricity and gas usage (kW)
    Class    - building type (16 types, e.g. Hospital, SmallOffice)
    theft    - Normal, or Theft1 to Theft6 (simulated theft types)

Structure found during the data checks (notebooks/01_data_quality.ipynb):
    - The file is 64 one-year recordings of 8,760 hours each:
      16 building types x 4 recordings per type.
    - 15 broken rows have building type "0".
    - The 4 recordings of a type are similar but not copies.

Cleaning steps, in order:
    1. Load, reading usage columns as float32 to halve memory.
    2. Drop the row-number column and the broken rows.
    3. Give each row a recording ID (1-64) and its hour (0-8759).
    4. Create a 0/1 theft label and keep the theft type.
    5. Shorten the column names.
    6. Split by recording: for each building type, some recordings go to
       the test set and the rest to training. No recording is in both.
"""

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760
ROW_NUMBER_COL = "0"
TYPE_COL_RAW = "Class"
LABEL_COL_RAW = "theft"

# Short, code-friendly names for the raw columns.
RENAME = {
    "Electricity:Facility [kW](Hourly)": "elec_facility_kw",
    "Fans:Electricity [kW](Hourly)": "elec_fans_kw",
    "Cooling:Electricity [kW](Hourly)": "elec_cooling_kw",
    "Heating:Electricity [kW](Hourly)": "elec_heating_kw",
    "InteriorLights:Electricity [kW](Hourly)": "elec_lights_kw",
    "InteriorEquipment:Electricity [kW](Hourly)": "elec_equipment_kw",
    "Gas:Facility [kW](Hourly)": "gas_facility_kw",
    "Heating:Gas [kW](Hourly)": "gas_heating_kw",
    "InteriorEquipment:Gas [kW](Hourly)": "gas_equipment_kw",
    "Water Heater:WaterSystems:Gas [kW](Hourly)": "gas_water_heater_kw",
    TYPE_COL_RAW: "building_type",
}
USAGE_COLS = [name for raw, name in RENAME.items() if raw != TYPE_COL_RAW]


def load_oedi(csv_path):
    """Read the raw OEDI file.

    Memory optimization: the 10 usage columns are read as float32
    (4 bytes) instead of the default float64 (8 bytes), halving their
    memory. Text columns are read as strings, so the broken building
    type "0" stays text and can be found.

    Returns:
        DataFrame with the raw columns.
    """
    dtypes = {raw: np.float32 for raw in RENAME if raw != TYPE_COL_RAW}
    dtypes[TYPE_COL_RAW] = str
    dtypes[LABEL_COL_RAW] = str
    return pd.read_csv(csv_path, dtype=dtypes)


def add_recording_ids(df):
    """Number each one-year recording and each hour inside it.

    Rows of one recording sit next to each other in the file, so a new
    recording starts wherever the building type changes.

    Raises:
        ValueError if any recording is not exactly one year (8,760 hours),
        which would mean the file is not structured as expected.
    """
    df = df.copy()
    new_block = df["building_type"] != df["building_type"].shift()
    df["recording_id"] = new_block.cumsum().astype(np.int16)
    df["hour"] = df.groupby("recording_id").cumcount().astype(np.int16)

    sizes = df.groupby("recording_id").size()
    wrong = sizes[sizes != HOURS_PER_YEAR]
    if len(wrong) > 0:
        raise ValueError(
            f"Expected every recording to have {HOURS_PER_YEAR} hours, "
            f"but these do not: {wrong.to_dict()}"
        )
    return df


def clean_oedi(raw):
    """Run all cleaning steps on the raw OEDI data.

    Args:
        raw: DataFrame from load_oedi().

    Returns:
        (clean, report): the cleaned DataFrame and a dict of numbers
        describing what each step did, for the data-quality summary.
    """
    report = {"rows_raw": len(raw),
              "missing_values_raw": int(raw.isnull().to_numpy().sum())}

    # Step 2: drop the row-number column and broken rows (building type "0").
    df = raw.drop(columns=[ROW_NUMBER_COL])
    broken = df[TYPE_COL_RAW] == "0"
    report["rows_removed_broken"] = int(broken.sum())
    df = df[~broken].reset_index(drop=True)

    # Step 5 (done early so later steps use the short names).
    df = df.rename(columns=RENAME)

    # Step 3: recording ID and hour.
    df = add_recording_ids(df)

    # Step 4: binary label (1 = any theft type) plus the theft type itself,
    # which is kept for error analysis.
    # The raw column is also called "theft", so rename it first, then
    # create the new 0/1 column.
    df = df.rename(columns={LABEL_COL_RAW: "theft_type"})
    df["theft"] = (df["theft_type"] != "Normal").astype(np.int8)

    # Put identifying columns first, then usage, then labels.
    df = df[["recording_id", "building_type", "hour"] + USAGE_COLS + ["theft", "theft_type"]]

    report["rows_clean"] = len(df)
    report["recordings"] = int(df["recording_id"].nunique())
    report["building_types"] = int(df["building_type"].nunique())
    report["theft_share"] = float(df["theft"].mean())
    return df, report


def split_oedi(clean, test_recordings_per_type, seed):
    """Split by recording: no recording appears in both sets.

    For every building type, `test_recordings_per_type` of its recordings
    are picked at random (using the seed) for the test set; the others go
    to training. This way:
      - every building type is seen during training;
      - test hours come from recordings the model never saw, so
        neighbouring hours of the same theft period cannot end up on
        both sides (which would inflate the scores).

    Args:
        clean: DataFrame from clean_oedi().
        test_recordings_per_type: e.g. 1 (1 of 4 = 25% test).
        seed: random seed from config.yaml.

    Returns:
        (train, test) DataFrames.
    """
    rng = np.random.default_rng(seed)
    recordings = clean.groupby("building_type")["recording_id"].unique()

    test_ids = []
    # Sorted, so the same seed always picks the same recordings.
    for building_type in sorted(recordings.index):
        ids = np.sort(recordings[building_type])
        test_ids.extend(rng.choice(ids, size=test_recordings_per_type, replace=False))

    is_test = clean["recording_id"].isin(test_ids)
    return clean[~is_test].reset_index(drop=True), clean[is_test].reset_index(drop=True)
