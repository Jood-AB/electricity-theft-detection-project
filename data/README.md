# Data

The datasets are **not stored in this repository**: they are large and
belong to their original publishers. Each member downloads them once and
places them in the folders below. The `.gitignore` file keeps them out of
GitHub.

## 1. SGCC (State Grid Corporation of China)

- **Source:** https://github.com/henryRDlab/ElectricityTheftDetection
- **Contents (as described in our proposal):** about 42,372 customers,
  1,034 daily electricity readings per customer, and a label showing
  whether the customer is flagged for theft.
- **Where to put it:** download the data file from the repository above,
  unzip it if needed, and place it in:

  ```
  data/raw/sgcc/
  ```

## 2. OEDI-based Theft Detection in Smart Grid Environment

- **Source:** https://data.mendeley.com/datasets/c3c7329tjj/1
- **Contents (as described in our proposal):** normal consumption patterns
  plus six simulated theft scenarios, based on US data.
- **Where to put it:** download the files from the page above and place
  them in:

  ```
  data/raw/oedi/
  ```

## Folder layout after download and cleaning

```
data/
├── raw/                 # original downloads (never edited by hand)
│   ├── sgcc/
│   └── oedi/
└── processed/           # created by the cleaning code (PR 2)
    ├── sgcc/            # cleaned SGCC data, train and test sets
    └── oedi/            # cleaned OEDI data, train and test sets
```

Exact file names and the cleaning steps will be documented here in PR 2,
once the loaders are written.

## Using the data in Google Colab (EDA only)

The EDA notebooks run in Google Colab. Upload the **processed** files to
Google Drive and read them from there, so the plots use the same cleaned
data as the models. See `notebooks/README.md`.
