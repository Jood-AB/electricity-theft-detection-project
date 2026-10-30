# Electricity Theft Detection from Smart Meter Data

CS316 (Introduction to AI and Data Science), Term 261, group project.

## Overview

Electricity utilities lose revenue when some electricity use is not
recorded or billed. This project tests whether supervised machine
learning can find consumption patterns linked to **potential** theft in
historical smart-meter data.

We train and compare three classifiers:

- Logistic Regression
- Decision Tree
- Random Forest

on two datasets, **processed and evaluated separately**:

- **SGCC**: real smart-meter data from the State Grid Corporation of China
- **OEDI-based**: US consumption data with simulated theft scenarios

## Sustainability

This project relates to **UN Sustainable Development Goal 7: Affordable and
Clean Energy**. Reducing unbilled electricity loss can support fairer
pricing and better planning of energy resources.

## Repository structure

```
electricity-theft-detection-project/
├── README.md            # this file
├── requirements.txt     # Python libraries to install
├── config.yaml          # shared settings: random seed, test size, folder paths
├── data/                # datasets (downloaded locally, not stored on GitHub)
│   └── README.md        # how to download each dataset, cleaning steps, citations
├── src/                 # all project code
│   ├── config.py        # reads config.yaml
│   ├── utils.py         # shared helpers, e.g. set_seed()
│   ├── data_sgcc.py     # loads and cleans SGCC, splits by customer
│   ├── data_oedi.py     # loads and cleans OEDI, splits by recording
│   ├── prepare_data.py  # runs both cleanings and saves train/test sets
│   ├── features.py      # builds the model features for SGCC and OEDI
│   ├── tuning.py        # hyperparameter tuning with cross-validation
│   ├── efficiency.py    # memory and runtime measurements, before/after
│   └── error_analysis.py  # false positives/negatives and feature importance
├── notebooks/           # notebooks (Google Colab)
│   ├── README.md        # how to use Colab with this project
│   ├── 01_data_quality.ipynb  # data checks behind each cleaning decision
│   └── 02_eda.ipynb           # EDA: class balance, missing data, trends, normal vs theft
└── tests/               # automated checks that the code works
    ├── test_setup.py      # config and seed
    ├── test_data_sgcc.py  # SGCC cleaning steps
    ├── test_data_oedi.py  # OEDI cleaning steps
    ├── test_features.py   # feature functions
    ├── conftest.py        # small made-up data shared by the tests below
    ├── test_tuning.py     # tuning and the efficiency changes
    └── test_error_analysis.py  # error analysis tables and importance
```

## Setup (local, for running the project code)

You need **Python 3.12** and **Git**.

1. **Download the project:**

   ```
   git clone https://github.com/Jood-AB/electricity-theft-detection-project.git
   cd electricity-theft-detection-project
   ```

2. **Create a virtual environment** (a private set of libraries for this
   project only):

   ```
   python -m venv .venv
   ```

3. **Turn it on:**

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Mac/Linux: `source .venv/bin/activate`

4. **Install the libraries:**

   ```
   pip install -r requirements.txt
   ```

5. **Download the datasets** following [data/README.md](data/README.md).

6. **Check that everything works:**

   ```
   python -m pytest
   ```

## Google Colab (EDA only)

The exploratory data analysis notebooks run in Google Colab. See
[notebooks/README.md](notebooks/README.md).

## How to run

1. **Clean the data and create the train/test sets** (about 1 minute):

   ```
   python -m src.prepare_data
   ```

   Creates `data/processed/sgcc/` and `data/processed/oedi/` (train.csv,
   test.csv) and a summary in `outputs/data_quality.md`. See
   [data/README.md](data/README.md) for the cleaning steps and
   [notebooks/01_data_quality.ipynb](notebooks/01_data_quality.ipynb) for
   the checks behind them.

2. **Build the features** (inside your own script or notebook):

   ```python
   from src import features

   X_sgcc = features.sgcc_features(sgcc_train)   # one row per customer
   y_sgcc = sgcc_train["FLAG"]

   X_oedi = features.oedi_features(oedi_train)   # one row per hour
   y_oedi = oedi_train["theft"]
   ```

   The functions return only feature columns: IDs (`CONS_NO`,
   `recording_id`) and labels (`FLAG`, `theft`, `theft_type`) are never
   included. Nothing is fitted on the data, so the same functions are
   applied to the test sets. See the docstrings in `src/features.py` for
   what each feature means.

3. **Train the baseline models:**

   ```
   python -m src.models           # cross-validation on the training data, both datasets
   python -m src.models --test    # final test-set scores (run once, at the end)
   ```

   Results are saved in `outputs/results/` and `outputs/figures/`.

4. **Tune the models** (about 40 minutes on a laptop; uses the training data only):

   ```
   python -m src.tuning           # cross-validated search, both datasets
   python -m src.tuning --test    # final test-set scores of the tuned models (run once, at the end)
   ```

   Saves the scores of the tuned models (`tuned_cv_<dataset>.csv`),
   every setting tried (`tuning_search_<dataset>.csv`), the search settings
   and seed (`tuning_settings_<dataset>.json`) and a Decision Tree
   validation curve in `outputs/figures/`.

5. **Measure memory and runtime** (about 3 minutes):

   ```
   python -m src.efficiency
   ```

   Saves before/after numbers for each optimization in
   `outputs/results/efficiency.csv` and a chart in `outputs/figures/`.

6. **Analyse the errors of the tuned Random Forest** (about 5 minutes, after step 4):

   ```
   python -m src.error_analysis          # cross-validated predictions on the training data
   python -m src.error_analysis --test   # test-set errors (only after the final test run)
   ```

   Saves who is wrongly flagged (false positives) and who is missed (false
   negatives), OEDI errors by theft type and building type, the theft share
   among the highest-risk rows, and permutation feature importance
   (`errors_*.csv`, `importance_*.csv` and figures). Importance shows what
   the model relies on, not what causes theft.

More steps will be added as each part of the project is completed.

## Team rules

- **One shared seed.** Always use `random_seed` from `config.yaml`, including
  as `random_state=` in scikit-learn functions.
- **Test set is used once.** Tune and compare models with cross-validation on
  the training data only. Use the test set only for the final results.
- **No customer in both sets.** Training and test sets never share a customer.
- **Datasets stay separate.** SGCC and OEDI are never mixed together.
- **Comment your code** so others can follow it.
- **Every change goes through a pull request** and is reviewed by another member.

## Team

| Member | Role |
|---|---|
| Jood | Data pipeline and reproducibility |
| Hessa | EDA and feature engineering |
| Sadeem | Baseline models, evaluation and integration |
| Asma | Tuning, efficiency and interpretation |

## Limitations

- The model flags **potential** theft. A flag is not proof of wrongdoing.
- The US theft cases are **simulated**, not real.
- Results on Chinese and US data may not apply to other countries,
  including Saudi Arabia. The model has not been tested there.
- This is an offline research project, not a live monitoring system.

## Use of AI

AI assistance was used for parts of the code and project setup. Details
will be given in the report's Use of AI statement.
