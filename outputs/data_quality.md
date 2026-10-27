# Data quality summary

Created by `python -m src.prepare_data`.

## SGCC

- Customers in raw file: 42,372; days: 1,034
- Missing readings in raw file: 25.6%
- Days added to complete the date range: 2016-09-18
- Customers removed (no readings at all): 5
- Readings capped by the IQR rule: 377,692 (from 25,165 customers)
- Customers skipped by the IQR rule (IQR = 0): 394
- Missing readings after cleaning: 0
- Clean data: 42,367 customers x 1,035 days

| Set | Customers | Theft share |
|---|---|---|
| Train | 33,893 | 8.5% |
| Test | 8,474 | 8.5% |

## OEDI

- Rows (hours) in raw file: 560,655; missing values: 0
- Broken rows removed (building type "0"): 15
- Clean data: 560,640 hours = 64 one-year recordings of 16 building types
- Theft share (all hours): 40.8%
- Test recordings: 16 (IDs [1, 7, 9, 10, 20, 21, 35, 40, 43, 45, 47, 50, 54, 60, 62, 64])

| Set | Hours | Theft share |
|---|---|---|
| Train | 420,480 | 40.8% |
| Test | 140,160 | 40.9% |
