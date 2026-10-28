"""Checks for the error analysis (PR 8), using small made-up data."""

import numpy as np
import pandas as pd
import pytest

from src import error_analysis as ea
from src import models, tuning


def test_outcome_labels():
    y = pd.Series([0, 0, 1, 1])
    got = ea.outcome_labels(y, [0.1, 0.9, 0.2, 0.8], threshold=0.5)
    assert got.tolist() == ["TN", "FP", "FN", "TP"]


def test_summary_rates():
    outcome = pd.Series(["TN"] * 6 + ["FP"] * 2 + ["FN"] * 1 + ["TP"] * 3)
    s = ea.summary_table("sgcc", outcome, 0.5, "cv").iloc[0]
    assert (s["TN"], s["FP"], s["FN"], s["TP"]) == (6, 2, 1, 3)
    assert s["false_positive_rate"] == 2 / 8
    assert s["false_negative_rate"] == 1 / 4
    assert s["false_alarms_among_flags"] == 2 / 5


def test_top_k_counts_theft_among_highest_scores():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table = ea.top_k_table(y, prob, shares=[0.1, 0.3]).set_index("top_share")
    assert table.loc[0.1, "precision"] == 1.0        # top 1 row is theft
    assert table.loc[0.3, "theft_found"] == 2         # top 3 rows hold both thefts
    assert table.loc[0.3, "recall"] == 1.0
    assert table.loc[0.1, "base_rate"] == 0.2


def test_sgcc_daily_readings_are_one_importance_group():
    columns = ["2014-01-01", "2014-01-02", "mean_use", "missing_share"]
    groups = ea.feature_groups("sgcc", columns)
    assert groups["daily readings (2 days)"] == ["2014-01-01", "2014-01-02"]
    assert groups["mean_use"] == ["mean_use"]
    assert ea.feature_groups("oedi", ["hour", "elec_facility_kw"]) == {
        "hour": ["hour"], "elec_facility_kw": ["elec_facility_kw"]}


def test_shuffling_a_useless_feature_changes_nothing():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.random(400), "noise": np.zeros(400)})
    y = pd.Series((X["signal"] > 0.5).astype(int))
    forest = models.make_models(42)["Random Forest"].fit(X, y)
    assert ea.permutation_drop(forest, X, y, ["noise"], rng) == 0
    assert ea.permutation_drop(forest, X, y, ["signal"], rng) > 0.3


def test_oedi_breakdowns():
    info = pd.DataFrame({"building_type": ["a", "a", "b", "b", "b"],
                         "theft_type": ["Normal", "Theft1", "Normal", "Normal", "Theft2"],
                         "theft": [0, 1, 0, 0, 1]})
    outcome = pd.Series(["FP", "TP", "TN", "FP", "FN"])
    by_type, by_building = ea.oedi_breakdowns(info, outcome)
    by_type = by_type.set_index("theft_type")["share_flagged"]
    assert by_type["Theft1"] == 1.0 and by_type["Theft2"] == 0.0
    assert by_type["Normal"] == 2 / 3
    rates = by_building.set_index("building_type")["false_positive_rate"]
    assert rates["a"] == 1.0 and rates["b"] == 0.5


def test_missing_tuned_model_gives_a_clear_message(featured_project):
    with pytest.raises(FileNotFoundError, match="src.tuning"):
        ea.load_saved("sgcc", "Random Forest")


@pytest.mark.parametrize("dataset", ["sgcc", "oedi"])
def test_analyse_runs_end_to_end(featured_project, monkeypatch, dataset):
    monkeypatch.setattr(tuning, "N_ITER", 2)
    monkeypatch.setattr(tuning, "CURVE_DEPTHS", [2, 3])
    data = featured_project / "data"
    tuning.tune(dataset, seed=42, data_dir=data)

    summary, importance = ea.analyse(dataset, seed=42, data_dir=data)
    counts = summary.iloc[0][["TN", "FP", "FN", "TP"]].sum()
    assert counts == len(pd.read_csv(data / dataset / "train.csv"))
    assert importance["pr_auc_drop"].notnull().all()

    results, figures = featured_project / "results", featured_project / "figures"
    assert (results / f"importance_{dataset}.csv").exists()
    assert (figures / f"errors_{dataset}_cv_scores.png").exists()
    if dataset == "sgcc":
        groups = pd.read_csv(results / "errors_sgcc_cv_groups.csv")
        assert groups["outcome"].tolist() == ["TN", "FP", "FN", "TP"]
    else:
        assert (results / "errors_oedi_cv_by_theft_type.csv").exists()

    # Test-set analysis: same tables, no importance (no refitting on test).
    summary, importance = ea.analyse(dataset, seed=42, use_test=True, data_dir=data)
    assert importance is None
    assert (results / f"errors_{dataset}_test_summary.csv").exists()
