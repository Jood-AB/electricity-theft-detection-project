"""Memory and runtime measurements (PR 7): before/after numbers for the report.

Each step of the project that is slow or uses a lot of memory is measured
twice: the simple version ("before") and the improved version that the
project code uses ("after"). Both versions must give the same result; the
script checks this before saving any number.

Improvements measured:
    1-2. Smaller number types and batch processing: read readings as
         float32 (4 bytes) instead of the default float64 (8 bytes), a few
         thousand rows at a time, for both datasets.
    3.   Fewer copies and batch processing: count SGCC sudden drops with
         NumPy on batches of customers instead of pandas rolling() on a
         transposed copy of the whole table.
    4.   Fewer copies: give the Random Forest float32 data. scikit-learn
         trees work in float32, so float64 input is first copied.
    5.   Batch processing: predict theft probabilities for a few thousand
         rows at a time instead of all rows at once.
    6.   Fewer columns: summarise SGCC's 1,035 daily readings with PCA
         (course topic 7.2) before the Random Forest. This one changes the
         model, so its PR-AUC is reported next to the time saved.

Only the TRAINING files are used; the test set is never touched.

How memory is measured: Python's tracemalloc records the highest amount of
memory allocated while a step runs (NumPy and pandas arrays included).
Time is measured in a separate run without tracemalloc, which slows code
down. Times depend on the computer, so compare before and after on the
same machine rather than the exact seconds.

Run from the project folder (after python -m src.prepare_data):
    python -m src.efficiency

Outputs:
    outputs/results/efficiency.csv
    outputs/figures/efficiency_before_after.png
"""

import time
import tracemalloc

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import features, models
from src.data_sgcc import ID_COL, LABEL_COL
from src.utils import READ_CHUNK_ROWS, read_csv_in_chunks, set_seed

matplotlib.use("Agg")  # draw without a screen, so it also works in scripts
import matplotlib.pyplot as plt  # noqa: E402

# Rows predicted at a time in step 5. Smaller batches use even less memory
# but are slower (20,000 rows: 16 MB peak but twice the time).
PREDICT_BATCH_ROWS = 100_000
# Number of PCA components kept for the daily readings in step 6.
PCA_COMPONENTS = 20


def measure(func):
    """Run func twice: once for the time, once for the peak memory.

    Returns:
        (result, seconds, peak_mb)
    """
    start = time.perf_counter()
    result = func()
    seconds = time.perf_counter() - start

    tracemalloc.start()
    func()
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return result, seconds, peak_mb


def frame_mb(df):
    """Memory held by a DataFrame, in MB (text columns included)."""
    return df.memory_usage(deep=True).sum() / 1e6


# ------------------------------------------------------------ before versions

def count_sudden_drops_pandas(readings, window=features.DROP_WINDOW_DAYS,
                              ratio=features.DROP_RATIO):
    """The first version of features.count_sudden_drops (PR 4), kept here
    only to measure it. It transposes the table, so pandas makes several
    full copies of all readings."""
    previous_avg = readings.T.rolling(window).mean().shift(1).T
    low = (readings < ratio * previous_avg) & (previous_avg > 0)
    previous_low = low.shift(1, axis=1, fill_value=False)
    return (low & ~previous_low).sum(axis=1)


def predict_in_batches(model, X, batch_rows=PREDICT_BATCH_ROWS):
    """Theft probabilities for X, computed batch_rows rows at a time."""
    parts = [model.predict_proba(X.iloc[i:i + batch_rows])[:, 1]
             for i in range(0, len(X), batch_rows)]
    return np.concatenate(parts)


# ------------------------------------------------------------------- steps

def step_loading(dataset, path, float_cols):
    """Steps 1-2: read a processed training file three ways.

    before:  pandas default (float64).
    middle:  dtype=float32 only. The finished table is half the size, but
             the reader's peak memory is not lower (it is higher for SGCC).
    after:   dtype=float32, read in chunks of rows (src/utils.py).
    """
    dtype = {c: np.float32 for c in float_cols}
    before, s0, p0 = measure(lambda: pd.read_csv(path))
    middle, s1, p1 = measure(lambda: pd.read_csv(path, dtype=dtype))
    after, s2, p2 = measure(lambda: read_csv_in_chunks(path, dtype))
    # Same values, only stored with fewer bytes.
    pd.testing.assert_frame_equal(after, middle)
    np.testing.assert_allclose(after[float_cols].to_numpy(), before[float_cols].to_numpy(),
                               rtol=1e-6, atol=1e-6)
    step = f"Load {dataset.upper()} training file"
    return [
        row(step, "before", "float64 (pandas default)", s0, p0, frame_mb(before)),
        row(step, "middle", "float32 readings, whole file at once", s1, p1, frame_mb(middle)),
        row(step, "after", f"float32 readings, {READ_CHUNK_ROWS:,} rows per chunk",
            s2, p2, frame_mb(after)),
    ], after


def step_sudden_drops(sgcc):
    """Step 3: count SGCC sudden drops, pandas vs batched NumPy."""
    readings = sgcc[features.sgcc_day_columns(sgcc)]
    before, s0, p0 = measure(lambda: count_sudden_drops_pandas(readings))
    after, s1, p1 = measure(lambda: features.count_sudden_drops(readings))
    assert (before.to_numpy() == after.to_numpy()).all(), "drop counts differ"
    step = "SGCC sudden-drop feature"
    return [
        row(step, "before", "pandas rolling on a transposed copy", s0, p0),
        row(step, "after", f"NumPy, {features.DROP_BATCH_SIZE:,} customers per batch", s1, p1),
    ]


def step_forest_dtype(X, y, seed):
    """Step 4: fit a Random Forest on float64 vs float32 data."""
    def fit(data):
        return models.make_models(seed)["Random Forest"].fit(data, y)

    X64 = X.astype(np.float64)
    before, s0, p0 = measure(lambda: fit(X64))
    after, s1, p1 = measure(lambda: fit(X))
    # The trees only ever see float32 values, so both forests give the same
    # probabilities (up to tiny rounding from adding the trees in parallel).
    np.testing.assert_allclose(before.predict_proba(X64.iloc[:2000]),
                               after.predict_proba(X.iloc[:2000]), atol=1e-12)
    step = "Fit Random Forest on SGCC (1,043 columns)"
    return [
        row(step, "before", "float64 input (copied to float32 inside)", s0, p0),
        row(step, "after", "float32 input (no copy)", s1, p1),
    ], after


def step_batch_predict(model, X):
    """Step 5: predict probabilities for all rows at once vs in batches."""
    before, s0, p0 = measure(lambda: model.predict_proba(X)[:, 1])
    after, s1, p1 = measure(lambda: predict_in_batches(model, X))
    np.testing.assert_allclose(before, after)
    step = f"Predict {len(X):,} OEDI hours (Random Forest)"
    return [
        row(step, "before", "all rows at once", s0, p0),
        row(step, "after", f"{PREDICT_BATCH_ROWS:,} rows per batch", s1, p1),
    ]


def step_pca(X, y, folds, seed):
    """Step 6: Random Forest on all SGCC columns vs PCA-reduced readings.

    PCA is fitted inside each cross-validation fold (training part only),
    so no information leaks from the validation part.
    """
    day_cols = [c for c in X.columns if c[:4].isdigit()]   # named "YYYY-MM-DD"
    other_cols = [c for c in X.columns if c not in day_cols]
    forest = models.make_models(seed)["Random Forest"]
    reduced = make_pipeline(
        ColumnTransformer([("pca", make_pipeline(StandardScaler(), PCA(PCA_COMPONENTS,
                                                                       random_state=seed)),
                            day_cols)],
                          remainder="passthrough"),
        forest)

    def cv_pr_auc(model):
        prob = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
        return average_precision_score(y, prob)

    before, s0, p0 = measure(lambda: cv_pr_auc(forest))
    after, s1, p1 = measure(lambda: cv_pr_auc(reduced))
    step = "Random Forest 3-fold CV on SGCC"
    return [
        row(step, "before", f"all {X.shape[1]:,} columns", s0, p0, pr_auc=before),
        row(step, "after", f"PCA: {len(day_cols):,} readings -> {PCA_COMPONENTS} components "
                           f"+ {len(other_cols)} other columns", s1, p1, pr_auc=after),
    ]


def row(step, version, how, seconds, peak_mb, result_mb=None, pr_auc=None):
    """One line of the results table."""
    return {"step": step, "version": version, "how": how,
            "seconds": round(seconds, 2), "peak_mb": round(peak_mb, 1),
            "result_mb": None if result_mb is None else round(result_mb, 1),
            "cv_pr_auc": None if pr_auc is None else round(pr_auc, 3)}


def plot(results, path):
    """Bar charts of time and peak memory, before vs after, for each step."""
    steps = list(dict.fromkeys(results["step"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(steps) + 1.5), sharey=True)
    y = np.arange(len(steps))
    for ax, col, label in [(axes[0], "seconds", "Time (s)"), (axes[1], "peak_mb", "Peak memory (MB)")]:
        for version, color, shift in [("before", "#b0b0b0", -0.2), ("after", "#2a7ab9", 0.2)]:
            values = results[results["version"] == version].set_index("step").loc[steps, col]
            ax.barh(y + shift, values, height=0.4, color=color, label=version)
        ax.set_xlabel(label)
    axes[0].set_yticks(y, steps)
    axes[0].invert_yaxis()   # first step at the top, "before" bar above "after"
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))   # outside the bars
    fig.suptitle("Efficiency: before vs after")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    config = models.CONFIG
    seed = config["random_seed"]
    set_seed(seed)
    paths = config["paths"]
    rows = []

    print("1. Loading SGCC...")
    sgcc_path = paths["sgcc_processed"] / "train.csv"
    columns = pd.read_csv(sgcc_path, nrows=0).columns
    new, sgcc = step_loading("sgcc", sgcc_path, [c for c in columns if c not in (ID_COL, LABEL_COL)])
    rows += new

    print("2. Loading OEDI...")
    oedi_path = paths["oedi_processed"] / "train.csv"
    oedi_cols = pd.read_csv(oedi_path, nrows=0).columns
    new, _ = step_loading("oedi", oedi_path, [c for c in oedi_cols if c.endswith("_kw")])
    rows += new

    print("3. Sudden drops...")
    rows += step_sudden_drops(sgcc)

    print("4. Random Forest fit...")
    X, y, _ = models.load_xy("sgcc", "train", add_features=True)
    new, _ = step_forest_dtype(X, y, seed)
    rows += new

    print("5. Batch prediction...")
    Xo, yo, _ = models.load_xy("oedi", "train", add_features=True)
    forest = models.make_models(seed)["Random Forest"].fit(Xo, yo)
    rows += step_batch_predict(forest, Xo)

    print("6. PCA...")
    rows += step_pca(X, y, models.make_folds(y, None, seed), seed)

    results = pd.DataFrame(rows)
    models.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(models.RESULTS_DIR / "efficiency.csv", index=False)
    plot(results, models.FIGURES_DIR / "efficiency_before_after.png")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
