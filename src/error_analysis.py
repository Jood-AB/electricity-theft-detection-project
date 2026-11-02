"""Error analysis and interpretation (PR 8).

Looks at the mistakes of a tuned model (default: Random Forest, the best
model on both datasets) to understand WHO it gets wrong:
    false positive (FP): a normal customer/hour wrongly flagged as theft
    false negative (FN): a theft-labelled customer/hour the model missed
and which inputs the model relies on (feature importance).

Which predictions are used:
    default:  cross-validated predictions on the TRAINING data. Each row is
              predicted by a model that did not see it, with the same folds
              and threshold as in src/tuning.py. The test set is not touched.
    --test:   the saved tuned model on the TEST set. Only after the team's
              final test run, and never to change the model afterwards.

Feature importance (permutation importance): on each validation fold, the
values of one feature (or one group of features) are shuffled between rows
and we measure how much PR-AUC drops. A big drop means the model leans on
that feature. The 1,035 SGCC daily readings are shuffled together as one
group, because one day on its own says little. When two features carry the
same information (e.g. mean and maximum use), they share the importance, so
each may look less important than it is. That is why all engineered
features (from src/features.py) are also shuffled together as one group,
which shows what they add as a set. Important: this shows what the MODEL
uses, not what causes theft.

Limitation (OEDI): the most important OEDI columns are end-use readings
(equipment, lights, fans, water heater). A real utility usually sees only
the main meter, so in practice a model would have less to work with.

Run from the project folder (after python -m src.tuning):
    python -m src.error_analysis
    python -m src.error_analysis --dataset sgcc
    python -m src.error_analysis --test

Outputs (the suffix is "cv" or "test"):
    outputs/results/errors_<dataset>_<suffix>_summary.csv   counts and rates of each outcome
    outputs/results/errors_<dataset>_<suffix>_top_k.csv     theft share among the highest-risk rows
    outputs/results/errors_sgcc_<suffix>_groups.csv         typical feature values of TP/FP/FN/TN
    outputs/results/errors_oedi_<suffix>_by_theft_type.csv  share found of each simulated theft type
    outputs/results/errors_oedi_<suffix>_by_building.csv    false alarms per building type
    outputs/results/importance_<dataset>.csv                permutation importance
    outputs/figures/...                                     one figure per table
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from src import models
from src.tuning import TAG
from src.utils import set_seed

matplotlib.use("Agg")  # draw without a screen, so it also works in scripts
import matplotlib.pyplot as plt  # noqa: E402

OUTCOMES = ["TN", "FP", "FN", "TP"]
OUTCOME_NAMES = {"TN": "Normal, not flagged", "FP": "Normal, flagged (false alarm)",
                 "FN": "Theft, missed", "TP": "Theft, flagged"}
COLORS = {"TN": "#9e9e9e", "FP": "#e08a2c", "FN": "#c0392b", "TP": "#2a7ab9"}
# Times each feature is shuffled per fold (the average is reported).
N_REPEATS = 3
# Shares of the highest-risk rows checked in the "inspection list" table.
TOP_SHARES = [0.01, 0.05, 0.10, 0.20]
# SGCC features compared between outcomes (from src/features.py).
SGCC_PROFILE_FEATURES = ["mean_use", "relative_std", "zero_days", "sudden_drops",
                         "missing_share", "max_use"]
# OEDI columns made by features.oedi_features (the rest are raw readings).
OEDI_ENGINEERED = ["hour_of_day", "elec_vs_mean_24h", "elec_rel_std_24h"]


# ----------------------------------------------------------- predictions

def load_saved(dataset, model_name):
    """Load a tuned model saved by src/tuning.py."""
    path = models.MODELS_DIR / f"{dataset}_{TAG}_{models.slug(model_name)}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python -m src.tuning first.")
    return joblib.load(path)


def feature_groups(dataset, columns):
    """Columns shuffled together in permutation importance.

    SGCC: all daily readings form one group; every other column is its own
    group. OEDI: every column is its own group. On both datasets, all
    engineered features are also shuffled together as one extra group.
    """
    if dataset == "sgcc":
        days = [c for c in columns if c[:4].isdigit()]   # named "YYYY-MM-DD"
        # Every SGCC column that is not a day is engineered: missing_share
        # from the cleaning step, the rest from features.sgcc_features.
        engineered = [c for c in columns if c not in days]
        groups = {f"daily readings ({len(days):,} days)": days}
        groups.update({c: [c] for c in engineered})
    else:
        engineered = [c for c in columns if c in OEDI_ENGINEERED]
        groups = {c: [c] for c in columns}
    if engineered:
        groups[f"all engineered features ({len(engineered)})"] = engineered
    return groups


def permutation_drop(model, X_val, y_val, columns, rng, n_repeats=N_REPEATS):
    """Average PR-AUC drop when `columns` are shuffled together."""
    base = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    drops = []
    for _ in range(n_repeats):
        shuffled = X_val.copy()
        order = rng.permutation(len(X_val))
        # The same row order for every column of the group, so the group
        # keeps its own pattern but loses its link to the label.
        shuffled[columns] = X_val[columns].to_numpy()[order]
        prob = model.predict_proba(shuffled)[:, 1]
        drops.append(base - average_precision_score(y_val, prob))
    return float(np.mean(drops))


def cv_predictions_and_importance(dataset, saved, X, y, groups, seed):
    """Cross-validated probabilities and permutation importance.

    For each fold, a copy of the tuned model is trained on the other folds,
    predicts the held-out fold, and is then used for permutation importance
    on that same held-out fold. So both use only rows the model did not see.
    """
    rng = np.random.default_rng(seed)
    prob = np.zeros(len(y))
    named_groups = feature_groups(dataset, list(X.columns))
    drops = {name: [] for name in named_groups}
    for train_idx, val_idx in models.make_folds(y, groups, seed):
        model = clone(saved["model"]).fit(X.iloc[train_idx], y.iloc[train_idx])
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        prob[val_idx] = model.predict_proba(X_val)[:, 1]
        for name, cols in named_groups.items():
            drops[name].append(permutation_drop(model, X_val, y_val, cols, rng))

    importance = pd.DataFrame({
        "feature": list(drops),
        "pr_auc_drop": [np.mean(v) for v in drops.values()],
        "pr_auc_drop_std": [np.std(v) for v in drops.values()],
        "columns": [len(named_groups[n]) for n in drops],
    }).sort_values("pr_auc_drop", ascending=False)
    return prob, importance


def outcome_labels(y, prob, threshold):
    """TP / FP / FN / TN for every row."""
    pred = np.asarray(prob) >= threshold
    truth = np.asarray(y) == 1
    return pd.Series(np.select([truth & pred, ~truth & pred, truth & ~pred],
                               ["TP", "FP", "FN"], default="TN"), index=y.index)


# --------------------------------------------------------------- tables

def summary_table(dataset, outcome, threshold, suffix):
    """How many rows end up in each outcome, and the main rates."""
    counts = outcome.value_counts().reindex(OUTCOMES, fill_value=0)
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    return pd.DataFrame([{
        "dataset": dataset, "predictions": suffix, "threshold": threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        # Share of normal rows wrongly flagged.
        "false_positive_rate": fp / max(fp + tn, 1),
        # Share of theft rows missed.
        "false_negative_rate": fn / max(fn + tp, 1),
        # Of all flagged rows, the share that are false alarms.
        "false_alarms_among_flags": fp / max(fp + tp, 1),
    }])


def top_k_table(y, prob, shares=TOP_SHARES):
    """Theft share among the highest-risk rows.

    A utility could inspect only the top 1%, 5%, ... of accounts ranked by
    the model. This shows how many of those would be theft-labelled
    (precision) and what share of all theft-labelled rows they cover
    (recall), compared with picking rows at random (the base rate).
    """
    order = np.argsort(-np.asarray(prob), kind="stable")
    y_sorted = np.asarray(y)[order]
    rows = []
    for share in shares:
        k = max(1, int(round(share * len(y))))
        found = int(y_sorted[:k].sum())
        rows.append({"top_share": share, "rows_checked": k, "theft_found": found,
                     "precision": found / k, "recall": found / max(int(np.sum(y)), 1),
                     "base_rate": float(np.mean(y))})
    return pd.DataFrame(rows)


def sgcc_group_table(feats, outcome):
    """Median of each SGCC feature for TN, FP, FN and TP customers."""
    table = feats[SGCC_PROFILE_FEATURES].groupby(outcome).median().reindex(OUTCOMES)
    table.insert(0, "customers", outcome.value_counts().reindex(OUTCOMES, fill_value=0))
    return table.rename_axis("outcome").reset_index()


def oedi_breakdowns(info, outcome):
    """OEDI errors by simulated theft type and by building type."""
    flagged = outcome.isin(["TP", "FP"])
    by_type = (pd.DataFrame({"theft_type": info["theft_type"], "flagged": flagged})
               .groupby("theft_type")["flagged"].agg(hours="size", share_flagged="mean")
               .reset_index())
    # For theft types the flagged share is the recall; for "Normal" it is
    # the false positive rate.
    normal = info["theft"].to_numpy() == 0
    by_building = (pd.DataFrame({"building_type": info["building_type"][normal],
                                 "false_alarm": flagged[normal]})
                   .groupby("building_type")["false_alarm"]
                   .agg(normal_hours="size", false_positive_rate="mean")
                   .sort_values("false_positive_rate", ascending=False).reset_index())
    return by_type, by_building


# ---------------------------------------------------------------- figures

def save(fig, name):
    models.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(models.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scores(dataset, y, prob, threshold, suffix):
    """Histogram of theft probabilities for normal and theft rows."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bins = np.linspace(0, 1, 41)
    for label, name, color in [(0, "Normal", COLORS["TN"]), (1, "Theft", COLORS["TP"])]:
        ax.hist(np.asarray(prob)[np.asarray(y) == label], bins=bins, alpha=0.6,
                density=True, label=name, color=color)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Predicted theft probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset.upper()}: where normal and theft rows overlap")
    ax.legend()
    save(fig, f"errors_{dataset}_{suffix}_scores.png")


def plot_sgcc_features(feats, outcome, suffix):
    """Box plots of key SGCC features for each outcome."""
    names = ["mean_use", "relative_std", "zero_days", "sudden_drops"]
    fig, axes = plt.subplots(1, len(names), figsize=(13, 3.4))
    for ax, name in zip(axes, names):
        data = [feats.loc[outcome == o, name] for o in OUTCOMES]
        box = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch, o in zip(box["boxes"], OUTCOMES):
            patch.set_facecolor(COLORS[o])
        ax.set_xticks(range(1, 5), OUTCOMES)
        ax.set_title(name)
    fig.suptitle("SGCC customers by outcome (TN/FP: normal, FN/TP: theft-labelled; outliers hidden)",
                 y=1.03)
    save(fig, f"errors_sgcc_{suffix}_features.png")


def plot_sgcc_profiles(readings, outcome, suffix):
    """Average weekly consumption over time for each outcome group."""
    days = pd.to_datetime(readings.columns)
    week = np.arange(len(days)) // 7           # 7 days in a row = one week
    week_start = days[::7]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for o in OUTCOMES:
        mean_day = readings[(outcome == o).to_numpy()].mean(axis=0).to_numpy()
        weekly = pd.Series(mean_day).groupby(week).mean()
        ax.plot(week_start, weekly, color=COLORS[o], label=f"{o}: {OUTCOME_NAMES[o]}")
    ax.set_ylabel("Average daily use (kWh)")
    ax.set_title("SGCC: average consumption of each outcome group (weekly)")
    ax.legend(fontsize=8)
    save(fig, f"errors_sgcc_{suffix}_profiles.png")


def plot_oedi_breakdowns(by_type, by_building, suffix):
    """Recall per theft type and false positive rate per building type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    colors = [COLORS["FP"] if t == "Normal" else COLORS["TP"] for t in by_type["theft_type"]]
    axes[0].bar(by_type["theft_type"], by_type["share_flagged"], color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of hours flagged")
    axes[0].set_title("By label (theft types: recall; Normal: false alarms)")
    axes[1].barh(by_building["building_type"], by_building["false_positive_rate"],
                 color=COLORS["FP"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("False positive rate (normal hours flagged)")
    axes[1].set_title("False alarms by building type")
    save(fig, f"errors_oedi_{suffix}_breakdown.png")


def plot_importance(dataset, importance, model_name):
    """Horizontal bars of the PR-AUC drop for each feature (group)."""
    imp = importance.sort_values("pr_auc_drop")
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(imp) + 1.2))
    ax.barh(imp["feature"], imp["pr_auc_drop"], xerr=imp["pr_auc_drop_std"],
            color=COLORS["TP"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Drop in validation PR-AUC when shuffled")
    ax.set_title(f"{dataset.upper()}: what the tuned {model_name} relies on\n"
                 "(association with the label, not cause)")
    save(fig, f"importance_{dataset}.png")


# ------------------------------------------------------------------ main

def analyse(dataset, seed, model_name="Random Forest", use_test=False, data_dir=models.DATA_DIR):
    """Run the error analysis for one dataset and save tables and figures."""
    saved = load_saved(dataset, model_name)
    threshold = saved["threshold"]
    split = "test" if use_test else "train"
    suffix = "test" if use_test else "cv"
    X, y, groups = models.load_xy(dataset, split, data_dir, add_features=True)
    X = X[saved["features"]]

    importance = None
    if use_test:
        prob = saved["model"].predict_proba(X)[:, 1]
    else:
        prob, importance = cv_predictions_and_importance(dataset, saved, X, y, groups, seed)
    outcome = outcome_labels(y, prob, threshold)

    out = models.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary = summary_table(dataset, outcome, threshold, suffix)
    summary.to_csv(out / f"errors_{dataset}_{suffix}_summary.csv", index=False)
    top_k_table(y, prob).to_csv(out / f"errors_{dataset}_{suffix}_top_k.csv", index=False)
    plot_scores(dataset, y, prob, threshold, suffix)

    if dataset == "sgcc":
        feats = X[[c for c in X.columns if not c[:4].isdigit()]]
        sgcc_group_table(feats, outcome).to_csv(out / f"errors_sgcc_{suffix}_groups.csv",
                                                index=False)
        plot_sgcc_features(feats, outcome, suffix)
        day_cols = [c for c in X.columns if c[:4].isdigit()]
        plot_sgcc_profiles(X[day_cols], outcome, suffix)
    else:
        # Text columns left out of X; same file, so the rows line up.
        info = pd.read_csv(Path(data_dir) / "oedi" / f"{split}.csv",
                           usecols=["building_type", "theft_type", "theft"])
        assert len(info) == len(outcome) and (info["theft"].to_numpy() == y.to_numpy()).all(), \
            "OEDI file rows no longer line up with the predictions"
        by_type, by_building = oedi_breakdowns(info, outcome)
        by_type.to_csv(out / f"errors_oedi_{suffix}_by_theft_type.csv", index=False)
        by_building.to_csv(out / f"errors_oedi_{suffix}_by_building.csv", index=False)
        plot_oedi_breakdowns(by_type, by_building, suffix)

    if importance is not None:
        importance.to_csv(out / f"importance_{dataset}.csv", index=False)
        plot_importance(dataset, importance, model_name)

    print(f"\n{dataset.upper()} ({suffix}), {model_name}, threshold {threshold:.2f}")
    print(summary.round(3).to_string(index=False))
    if importance is not None:
        print(importance.round(4).head(8).to_string(index=False))
    return summary, importance


def main():
    parser = argparse.ArgumentParser(description="Error analysis of a tuned model.")
    parser.add_argument("--dataset", choices=["sgcc", "oedi", "all"], default="all")
    parser.add_argument("--model", default="Random Forest",
                        choices=["Logistic Regression", "Decision Tree", "Random Forest"])
    parser.add_argument("--test", action="store_true",
                        help="analyse test-set errors (only after the final test run)")
    args = parser.parse_args()

    seed = models.CONFIG["random_seed"]
    set_seed(seed)
    datasets = ["sgcc", "oedi"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        analyse(dataset, seed, args.model, args.test)
    print(f"\nResults saved in {models.RESULTS_DIR}/ and {models.FIGURES_DIR}/")


if __name__ == "__main__":
    main()
