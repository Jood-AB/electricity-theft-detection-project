# Data

The datasets are **not stored in this repository**: they are large and
belong to their original publishers. Each member downloads them once and
places them in the folders below. The `.gitignore` file keeps them out of
GitHub.

## 1. SGCC (State Grid Corporation of China)

- **Source:** https://github.com/henryRDlab/ElectricityTheftDetection
- **Download:** the data is split into three parts: `data.z01`, `data.z02`
  and `data.zip`. Download all three into `data/raw/sgcc/` and extract
  `data.zip` with 7-Zip (Windows' built-in unzip cannot open split zips).
- **Expected file:** `data/raw/sgcc/data.csv` (about 167 MB)
- **Contents:** 42,372 customers; columns `CONS_NO` (customer ID),
  `FLAG` (1 = theft, 0 = normal) and one column per day from 2014-01-01
  to 2016-10-31 (daily usage in kWh). About 8.5% of customers are
  labelled as theft.
- **Reference:** Z. Zheng, Y. Yang, X. Niu, H.-N. Dai and Y. Zhou,
  "Wide and Deep Convolutional Neural Networks for Electricity-Theft
  Detection to Secure Smart Grids," *IEEE Transactions on Industrial
  Informatics*, vol. 14, no. 4, pp. 1606-1615, 2018,
  doi: 10.1109/TII.2017.2785963.

## 2. OEDI-based Theft Detection in Smart Grid Environment

- **Source:** https://data.mendeley.com/datasets/c3c7329tjj/1
- **Download:** download the zip from the page into `data/raw/oedi/` and
  extract it.
- **Expected file:** `data/raw/oedi/df.csv` (about 61 MB)
- **Contents:** hourly electricity and gas usage (kW) for 16 building types
  (64 one-year recordings), based on US Open Energy Data Initiative (OEDI)
  data. Column `theft` is `Normal` or one of six simulated theft types
  (`Theft1` to `Theft6`); about 41% of hours are theft.
- **Licence:** CC BY 4.0
- **Reference:** S. Zidi, A. Mihoub, S. M. Qaisar, M. Krichen and
  Q. Abu Al-Haija, "Theft detection dataset for benchmarking and machine
  learning based classification in a smart grid environment," *Journal of
  King Saud University - Computer and Information Sciences*, 2022,
  doi: 10.1016/j.jksuci.2022.05.007. Dataset: Mendeley Data, V1,
  doi: 10.17632/c3c7329tjj.1.

## Cleaning (PR 2)

Run from the project folder:

```
python -m src.prepare_data
```

This creates the processed files and a summary in `outputs/data_quality.md`.
The checks behind each decision are in
[notebooks/01_data_quality.ipynb](../notebooks/01_data_quality.ipynb).

**SGCC** (`src/data_sgcc.py`):
1. Read daily readings as float32 (half the memory of float64).
2. Sort day columns by date and add the one missing day (2016-09-18).
3. Remove 5 customers with no readings.
4. Save each customer's share of missing days as `missing_share`.
5. Cap extreme high readings per customer: IQR rule, upper limit
   Q3 + 3 x IQR, computed on the customer's non-zero days.
6. Fill gaps: linear interpolation in the middle, backward fill at the
   start, forward fill at the end.
7. Split 80/20 by customer, stratified by `FLAG`, seed 42.

**OEDI** (`src/data_oedi.py`):
1. Read usage columns as float32.
2. Drop the row-number column `0` and 15 broken rows (building type `"0"`).
3. Add `recording_id` (1-64) and `hour` (0-8759).
4. Create a 0/1 `theft` label; keep the original as `theft_type`.
5. Shorten column names (e.g. `elec_facility_kw`).
6. Split by recording: for each building type, 1 of its 4 recordings goes
   to the test set (seed 42), 3 to training.

## Folder layout

```
data/
├── raw/                  # original downloads (never edited)
│   ├── sgcc/data.csv
│   └── oedi/df.csv
└── processed/            # created by python -m src.prepare_data
    ├── sgcc/train.csv, test.csv
    └── oedi/train.csv, test.csv
```

**Note:** CSV files do not remember that numbers were float32. When loading
the processed files, pass `dtype` to `pd.read_csv` (as the loaders in
`src/` do) to keep memory low.

## Using the data in Google Colab

Upload the files you need to Google Drive and read them from there. See
`notebooks/README.md`. EDA should use the processed **training** files, so
the plots match what the models see.
