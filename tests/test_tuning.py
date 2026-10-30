"""Checks for tuning and the efficiency changes (PR 7), using small made-up data."""

import json

import numpy as np
import pandas as pd
import pytest

from src import efficiency, features, models, tuning
from src.utils import read_csv_in_chunks


@pytest.fixture
def quick_search(monkeypatch):
    """Try fewer settings so the tests run in seconds."""
    monkeypatch.setattr(tuning, "N_ITER", 2)
    monkeypatch.setattr(tuning, "CURVE_DEPTHS", [2, 3])


# ------------------------------------------------------------- efficiency

def test_fast_sudden_drops_match_the_first_version():
    rng = np.random.default_rng(1)
    readings = pd.DataFrame(rng.uniform(0, 10, size=(25, 60)).astype(np.float32))
    readings.iloc[:, 20:23] = 0          # some zero days and sudden drops
    readings.iloc[3] = 0                 # a customer who always uses 0
    expected = efficiency.count_sudden_drops_pandas(readings)
    # A batch size that does not divide the number of customers, so the
    # last batch is shorter than the others.
    got = features.count_sudden_drops(readings, batch_size=7)
    assert got.tolist() == expected.tolist()
    assert got.index.equals(readings.index)


def test_reading_in_chunks_gives_the_same_table(tmp_path):
    df = pd.DataFrame({"id": [f"C{i}" for i in range(23)],
                       "use": np.linspace(0, 5, 23)})
    df.to_csv(tmp_path / "f.csv", index=False)
    dtype = {"use": np.float32}
    got = read_csv_in_chunks(tmp_path / "f.csv", dtype, chunk_rows=5)
    pd.testing.assert_frame_equal(got, pd.read_csv(tmp_path / "f.csv", dtype=dtype))


def test_batch_prediction_matches_all_at_once():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((250, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0.5).astype(int)
    forest = models.make_models(42)["Random Forest"].fit(X, y)
    np.testing.assert_allclose(efficiency.predict_in_batches(forest, X, batch_rows=40),
                               forest.predict_proba(X)[:, 1])


# ----------------------------------------------------------------- tuning

def test_search_space_depends_on_dataset():
    # SGCC has 1,043 columns, so the forest only tries small column shares.
    assert tuning.search_space("Random Forest", "sgcc")["max_features"] == ["sqrt", "log2"]
    assert 1.0 in tuning.search_space("Random Forest", "oedi")["max_features"]
    assert tuning.search_space("Decision Tree", "sgcc") == tuning.search_space("Decision Tree", "oedi")


@pytest.mark.parametrize("dataset", ["sgcc", "oedi"])
def test_tune_then_test_runs_end_to_end(featured_project, quick_search, dataset):
    data = featured_project / "data"
    cv = tuning.tune(dataset, seed=42, data_dir=data)
    assert list(cv["model"]) == ["Logistic Regression", "Decision Tree", "Random Forest"]
    assert cv["f1"].between(0, 1).all() and cv["pr_auc"].between(0, 1).all()

    results = featured_project / "results"
    with open(results / f"tuning_settings_{dataset}.json", encoding="utf-8") as f:
        settings = json.load(f)
    # Everything needed to repeat the search is recorded.
    assert settings["random_seed"] == 42
    for name, info in settings["models"].items():
        for param, value in info["best_params"].items():
            assert value in tuning.search_space(name, dataset)[param]
    searched = pd.read_csv(results / f"tuning_search_{dataset}.csv")
    assert set(searched["model"]) == set(cv["model"])
    assert (featured_project / "figures" / f"validation_curve_{dataset}_decision_tree.png").exists()

    test = tuning.test_tuned(dataset, data_dir=data)
    assert len(test) == 3
    assert (featured_project / "figures" / f"cm_{dataset}_tuned_random_forest.png").exists()


def test_tuning_never_reads_the_test_set(featured_project, quick_search, monkeypatch):
    splits = []
    original = models.load_xy

    def spy(dataset, split="train", *args, **kwargs):
        splits.append(split)
        return original(dataset, split, *args, **kwargs)

    monkeypatch.setattr(models, "load_xy", spy)
    tuning.tune("oedi", seed=42, data_dir=featured_project / "data")
    assert splits == ["train"]


def test_same_seed_gives_same_tuning(featured_project, quick_search):
    data = featured_project / "data"
    a = tuning.tune("sgcc", seed=42, data_dir=data)
    b = tuning.tune("sgcc", seed=42, data_dir=data)
    pd.testing.assert_frame_equal(a.drop(columns="seconds"), b.drop(columns="seconds"))
