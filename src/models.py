"""Baseline models (PR 5): Logistic Regression, Decision Tree, Random Forest.

The same three models are trained on SGCC and OEDI separately.

How it works for each dataset:
1. Load the processed training file made by `python -m src.prepare_data`.
2. Cross-validation on the training data gives each row a theft probability
   from a model that did not see that row.
3. Those probabilities are used to choose each model's threshold (best F1)
   and to report validation metrics. The test set is not touched.
4. Each model is then trained on all training data and saved.

Only with --test are the saved models scored on the test set. Team rule:
do this once, for the final results.

Run from the project folder:
    python -m src.models                    # both datasets, validation only
    python -m src.models --dataset sgcc     # one dataset
    python -m src.models --sample 2000      # quick run on a small sample (for testing the code only)
    python -m src.models --test             # final scores on the test set

When Member 2's engineered features are saved as train/test files (one
folder per dataset, same label and ID columns), run the same code on them:
    python -m src.models --data-dir data/features --tag features
"""

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.metrics import best_threshold, evaluate, plot_confusion_matrix

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

# For each dataset: the label column, columns that must NOT be used as
# features (IDs or the answer), and the column that groups rows for
# cross-validation (None = each row is its own customer).
DATASETS = {
    "sgcc": {"label": "FLAG", "drop": ["CONS_NO"], "group": None},
    "oedi": {"label": "theft", "drop": ["theft_type", "recording_id"],
             "group": "recording_id"},
}

N_FOLDS = 3  # 3 folds keeps Random Forest fast enough for Colab


def load_seed(config_path="config.yaml"):
    """Read the shared random seed from config.yaml (team rule)."""
    with open(config_path) as f:
        return yaml.safe_load(f).get("random_seed", 42)


def load_xy(dataset, split="train", data_dir=DATA_DIR):
    """Load a processed file and return features X, labels y and groups.

    Numbers are read as float32 to halve memory. Text columns (such as the
    OEDI building type) are left out of X.
    """
    info = DATASETS[dataset]
    path = Path(data_dir) / dataset / f"{split}.csv"

    # Look at the first rows to find the number columns, then read them as float32
    head = pd.read_csv(path, nrows=100)
    not_features = [info["label"]] + info["drop"]
    float_cols = [c for c in head.select_dtypes("number").columns if c not in not_features]
    df = pd.read_csv(path, dtype={c: "float32" for c in float_cols})

    X = df[float_cols]
    y = df[info["label"]].astype(int)
    groups = df[info["group"]] if info["group"] else None
    return X, y, groups


def make_models(seed):
    """Return the three baseline models with simple, fixed settings.

    class_weight="balanced" gives the rare theft class more weight, so the
    models do not just predict "normal" for everyone. Tuning is Member 4's job.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=10, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=seed),
    }


def make_folds(y, groups, seed):
    """Split training data into folds without leakage.

    SGCC: one row per customer, so stratified folds keep the theft share equal.
    OEDI: many hours per recording, so all hours of a recording stay in one fold.
    """
    if groups is None:
        return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(y, y))
    return list(GroupKFold(N_FOLDS).split(y, y, groups))


def sample_rows(X, y, groups, n, seed):
    """Keep about n rows for a quick test run (whole recordings for OEDI)."""
    if groups is None:
        keep = y.groupby(y, group_keys=False).apply(
            lambda part: part.sample(frac=min(1, n / len(y)), random_state=seed)).index
    else:
        ids = groups.drop_duplicates().sample(frac=1, random_state=seed)
        per_group = len(y) / ids.size
        keep_ids = ids.iloc[:max(N_FOLDS, round(n / per_group))]
        keep = groups[groups.isin(keep_ids)].index
    keep = sorted(keep)
    return X.loc[keep], y.loc[keep], None if groups is None else groups.loc[keep]


def train_baselines(dataset, seed, sample=None, data_dir=DATA_DIR, tag="raw"):
    """Cross-validate, choose thresholds, then fit and save each model."""
    X, y, groups = load_xy(dataset, "train", data_dir)
    if sample:
        X, y, groups = sample_rows(X, y, groups, sample, seed)
    print(f"\n{dataset.upper()}: {len(y):,} rows, {X.shape[1]} features, "
          f"theft share {y.mean():.3f}")

    folds = make_folds(y, groups, seed)
    rows = []
    for name, model in make_models(seed).items():
        start = time.time()
        # Probability for every training row, from a model that did not see it
        prob = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
        threshold = best_threshold(y, prob)
        scores = evaluate(y, prob, threshold)
        default = evaluate(y, prob, 0.5)

        model.fit(X, y)  # final model on all training data
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "threshold": threshold, "features": list(X.columns)},
                    MODELS_DIR / f"{dataset}_{tag}_{slug(name)}.joblib")

        rows.append({"dataset": dataset, "features": tag, "model": name,
                     "f1_at_0.5": default["f1"], **scores,
                     "seconds": round(time.time() - start, 1)})
        print(f"  {name}: threshold {threshold:.2f}, F1 {scores['f1']:.3f} "
              f"(F1 at 0.5: {default['f1']:.3f}), PR-AUC {scores['pr_auc']:.3f}")

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"baseline_cv_{dataset}_{tag}.csv", index=False)
    return results


def test_baselines(dataset, data_dir=DATA_DIR, tag="raw"):
    """Score the saved models on the test set. Use once, for final results."""
    X, y, _ = load_xy(dataset, "test", data_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in make_models(seed=0):
        saved = joblib.load(MODELS_DIR / f"{dataset}_{tag}_{slug(name)}.joblib")
        prob = saved["model"].predict_proba(X[saved["features"]])[:, 1]
        rows.append({"dataset": dataset, "features": tag, "model": name,
                     **evaluate(y, prob, saved["threshold"])})
        plot_confusion_matrix(y, prob, saved["threshold"], f"{dataset.upper()} {name}",
                              FIGURES_DIR / f"cm_{dataset}_{tag}_{slug(name)}.png")
    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"baseline_test_{dataset}_{tag}.csv", index=False)
    return results


def slug(name):
    """'Random Forest' -> 'random_forest' (for file names)."""
    return name.lower().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Train the baseline models.")
    parser.add_argument("--dataset", choices=["sgcc", "oedi", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None,
                        help="use about this many training rows (quick test run)")
    parser.add_argument("--test", action="store_true",
                        help="score saved models on the test set (final results only)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="folder with <dataset>/train.csv and test.csv")
    parser.add_argument("--tag", default="raw",
                        help="name for this feature set in output files, e.g. raw or features")
    args = parser.parse_args()

    datasets = ["sgcc", "oedi"] if args.dataset == "all" else [args.dataset]
    seed = load_seed()
    for dataset in datasets:
        if args.test:
            print(test_baselines(dataset, args.data_dir, args.tag).round(3).to_string(index=False))
        else:
            train_baselines(dataset, seed, args.sample, args.data_dir, args.tag)
    print(f"\nResults saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
