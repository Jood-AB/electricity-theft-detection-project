"""Hyperparameter tuning (PR 7): search for better settings for the three models.

The baseline models (src/models.py) use simple fixed settings. Here we try
other settings for each model and keep the best one, on SGCC and OEDI
separately. All three models use the raw readings plus Member 2's
engineered features (the best feature set in PR 6).

How it works for each dataset and model:
1. Search: try several settings with cross-validation on the TRAINING data
   only, using the same folds as the baselines (SGCC: stratified folds;
   OEDI: all hours of a recording stay in one fold). Settings are compared
   by PR-AUC, which does not depend on a threshold and suits rare theft.
2. The best settings get a theft probability for every training row from
   cross-validation, and the threshold with the best F1 is chosen from
   those probabilities (the same way as the baselines, so the two are
   compared fairly).
3. The tuned model is trained on all training data and saved.

The test set is not used, except with --test (team rule: once, at the end).

Caveat: the cross-validated scores of the tuned models are a little
optimistic, for two reasons. The threshold is chosen and scored on the same
predictions (as for the baselines), and the best setting is picked on the
same folds it is scored on, so part of a small gain over the baseline can be
luck from trying many settings. Small gains should be called "likely" until
the single --test run confirms them.

Everything needed to repeat the search is saved: the settings that were
tried, the number of random tries, the folds and the seed.

Run from the project folder:
    python -m src.tuning                    # both datasets
    python -m src.tuning --dataset oedi     # one dataset
    python -m src.tuning --sample 3000      # quick run on a small sample (testing the code only)
    python -m src.tuning --test             # final test-set scores of the tuned models

Outputs:
    outputs/results/tuned_cv_<dataset>.csv         scores of the tuned models (validation)
    outputs/results/tuning_search_<dataset>.csv    every setting tried, with its scores
    outputs/results/tuning_settings_<dataset>.json search settings, seed and best settings
    outputs/figures/validation_curve_<dataset>_decision_tree.png
    outputs/models/<dataset>_tuned_<model>.joblib  (not on GitHub, too large)
"""

import argparse
import json
import time

import joblib
import matplotlib
import pandas as pd
from joblib import parallel_config
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_predict, validation_curve)

from src import models
from src.metrics import best_threshold, evaluate, plot_confusion_matrix
from src.utils import set_seed

matplotlib.use("Agg")  # draw without a screen, so it also works in scripts
import matplotlib.pyplot as plt  # noqa: E402

TAG = "tuned"
# PR-AUC (average precision): ranks settings by how well they put theft
# cases at the top, for any threshold.
SCORING = "average_precision"
# Number of random settings tried for the Decision Tree and Random Forest.
# Logistic Regression has one hyperparameter (C), so all its values are tried.
N_ITER = 12

# The settings tried for each model.
#   C (Logistic Regression): smaller C = simpler model (stronger penalty).
#   max_depth, min_samples_leaf: limit how detailed each tree can get. Deep
#       trees with tiny leaves fit the training data closely and can overfit.
#   n_estimators: number of trees in the forest.
#   max_features: how many columns each split may look at.
# SGCC has 1,043 columns, so "log2" (10) and "sqrt" (32) keep the forest
# fast. OEDI has only 14 columns, so larger shares are affordable there.
SEARCH_SPACES = {
    "Logistic Regression": {
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Decision Tree": {
        "max_depth": [4, 6, 8, 10, 15, 20, None],
        "min_samples_leaf": [1, 5, 20, 50, 100],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3, 10],
        "max_features": {"sgcc": ["sqrt", "log2"], "oedi": ["sqrt", 0.5, 1.0]},
    },
}

# Depths drawn in the validation curve (bias/variance picture).
CURVE_DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20, 25]


def search_space(model_name, dataset):
    """Return the settings to try for one model on one dataset."""
    space = {}
    for name, values in SEARCH_SPACES[model_name].items():
        space[name] = values[dataset] if isinstance(values, dict) else values
    return space


def make_search(model_name, model, space, folds, seed):
    """Build the search object for one model.

    Logistic Regression: every value is tried (grid search).
    Trees: N_ITER random combinations are tried (randomized search), which
    covers the space well at a fraction of the cost of trying all of them.

    The Random Forest already uses every processor core for its trees, so
    its settings are tried one after another. The other models are small,
    so several settings are tried at the same time (in threads, see tune()).
    """
    n_jobs = 1 if model_name == "Random Forest" else -1
    common = dict(scoring=SCORING, cv=folds, n_jobs=n_jobs, refit=False,
                  return_train_score=True)
    if model_name == "Logistic Regression":
        return GridSearchCV(model, space, **common)
    return RandomizedSearchCV(model, space, n_iter=N_ITER, random_state=seed, **common)


def search_table(dataset, model_name, search):
    """Every setting tried, with its train and validation PR-AUC."""
    res = search.cv_results_
    return pd.DataFrame({
        "dataset": dataset,
        "model": model_name,
        "params": [json.dumps(p, default=str) for p in res["params"]],
        "val_pr_auc": res["mean_test_score"],
        "val_pr_auc_std": res["std_test_score"],
        "train_pr_auc": res["mean_train_score"],
        "fit_seconds": res["mean_fit_time"],
        "rank": res["rank_test_score"],
    }).sort_values("rank")


def tune(dataset, seed, sample=None, data_dir=models.DATA_DIR):
    """Search, choose thresholds, then fit and save each tuned model."""
    X, y, groups = models.load_xy(dataset, "train", data_dir, add_features=True)
    if sample:
        X, y, groups = models.sample_rows(X, y, groups, sample, seed)
    print(f"\n{dataset.upper()}: {len(y):,} rows, {X.shape[1]} features, "
          f"theft share {y.mean():.3f}")

    # Same folds as the baselines, so tuned and baseline scores are comparable.
    folds = models.make_folds(y, groups, seed)
    settings = {
        "dataset": dataset,
        "rows": int(len(y)),
        "columns": int(X.shape[1]),
        "feature_set": "raw readings + engineered features",
        "random_seed": seed,
        "cv": (f"GroupKFold({models.N_FOLDS}) by recording_id" if groups is not None
               else f"StratifiedKFold({models.N_FOLDS}, shuffle=True, random_state={seed})"),
        "scoring": f"{SCORING} (PR-AUC)",
        "n_iter_random_search": N_ITER,
        "threshold_rule": "highest F1 on cross-validated training predictions",
        "sample": sample,
        "models": {},
    }

    rows, tables = [], []
    for name, base in models.make_models(seed).items():
        start = time.time()
        space = search_space(name, dataset)
        search = make_search(name, base, space, folds, seed)
        # Threads share the training data, so it is not copied into a
        # separate process for each setting (fewer copies, less memory).
        with parallel_config(backend="threading"):
            search.fit(X, y)
        search_seconds = time.time() - start
        best = search.best_params_
        tables.append(search_table(dataset, name, search))

        # Probabilities from the best settings for every training row,
        # each from a model that did not see that row.
        model = clone(base).set_params(**best)
        prob = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
        threshold = best_threshold(y, prob)
        scores = evaluate(y, prob, threshold)
        default = evaluate(y, prob, 0.5)

        model.fit(X, y)  # final tuned model on all training data
        models.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "threshold": threshold, "features": list(X.columns),
                     "best_params": best},
                    models.MODELS_DIR / f"{dataset}_{TAG}_{models.slug(name)}.joblib")

        i = search.best_index_
        settings["models"][name] = {
            "search": type(search).__name__,
            "space": space,
            "candidates_tried": len(search.cv_results_["params"]),
            "best_params": best,
            "best_val_pr_auc": float(search.cv_results_["mean_test_score"][i]),
            "best_train_pr_auc": float(search.cv_results_["mean_train_score"][i]),
            "search_seconds": round(search_seconds, 1),
        }
        rows.append({"dataset": dataset, "features": "features", "model": name,
                     "version": TAG, "f1_at_0.5": default["f1"], **scores,
                     "seconds": round(time.time() - start, 1),
                     "best_params": json.dumps(best, default=str)})
        print(f"  {name}: best {best}\n"
              f"    threshold {threshold:.2f}, F1 {scores['f1']:.3f}, "
              f"PR-AUC {scores['pr_auc']:.3f} ({search_seconds:.0f} s search)")

    results = pd.DataFrame(rows)
    models.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(models.RESULTS_DIR / f"tuned_cv_{dataset}.csv", index=False)
    pd.concat(tables).to_csv(models.RESULTS_DIR / f"tuning_search_{dataset}.csv", index=False)
    with open(models.RESULTS_DIR / f"tuning_settings_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, default=str)

    plot_depth_curve(dataset, X, y, folds, seed)
    return results


def plot_depth_curve(dataset, X, y, folds, seed, depths=CURVE_DEPTHS):
    """Plot train and validation PR-AUC of the Decision Tree for each depth.

    A shallow tree is too simple for both (underfitting, high bias). A very
    deep tree keeps improving on the training folds but gets worse on the
    validation folds (overfitting, high variance). The best depth is where
    the validation score peaks.
    """
    tree = models.make_models(seed)["Decision Tree"]
    with parallel_config(backend="threading"):   # shares X, as in tune()
        train, val = validation_curve(tree, X, y, param_name="max_depth", param_range=depths,
                                      cv=folds, scoring=SCORING, n_jobs=-1)
    curve = pd.DataFrame({"dataset": dataset, "max_depth": depths,
                          "train_pr_auc": train.mean(axis=1),
                          "val_pr_auc": val.mean(axis=1), "val_pr_auc_std": val.std(axis=1)})
    curve.to_csv(models.RESULTS_DIR / f"validation_curve_{dataset}_decision_tree.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(depths, curve["train_pr_auc"], "o-", label="Training folds")
    ax.plot(depths, curve["val_pr_auc"], "s-", label="Validation folds")
    ax.fill_between(depths, curve["val_pr_auc"] - curve["val_pr_auc_std"],
                    curve["val_pr_auc"] + curve["val_pr_auc_std"], alpha=0.2)
    best = curve.loc[curve["val_pr_auc"].idxmax(), "max_depth"]
    ax.axvline(best, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Decision Tree max_depth")
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"{dataset.upper()}: Decision Tree depth (best validation depth {best})")
    ax.legend()
    models.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(models.FIGURES_DIR / f"validation_curve_{dataset}_decision_tree.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return curve


def test_tuned(dataset, data_dir=models.DATA_DIR):
    """Score the saved tuned models on the test set. Use once, for final results."""
    X, y, _ = models.load_xy(dataset, "test", data_dir, add_features=True)
    models.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in models.make_models(seed=0):
        saved = joblib.load(models.MODELS_DIR / f"{dataset}_{TAG}_{models.slug(name)}.joblib")
        prob = saved["model"].predict_proba(X[saved["features"]])[:, 1]
        rows.append({"dataset": dataset, "features": "features", "model": name,
                     "version": TAG, **evaluate(y, prob, saved["threshold"])})
        plot_confusion_matrix(y, prob, saved["threshold"], f"{dataset.upper()} tuned {name}",
                              models.FIGURES_DIR / f"cm_{dataset}_{TAG}_{models.slug(name)}.png")
    results = pd.DataFrame(rows)
    models.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(models.RESULTS_DIR / f"tuned_test_{dataset}.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Tune the three models.")
    parser.add_argument("--dataset", choices=["sgcc", "oedi", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None,
                        help="use about this many training rows (quick test run)")
    parser.add_argument("--test", action="store_true",
                        help="score the tuned models on the test set (final results only)")
    args = parser.parse_args()

    datasets = ["sgcc", "oedi"] if args.dataset == "all" else [args.dataset]
    seed = models.CONFIG["random_seed"]
    set_seed(seed)
    for dataset in datasets:
        if args.test:
            print(test_tuned(dataset).round(3).to_string(index=False))
        else:
            tune(dataset, seed, args.sample)
    print(f"\nResults saved in {models.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
