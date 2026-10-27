"""Checks for src/metrics.py and src/models.py, using small made-up data."""

import numpy as np
import pandas as pd
import pytest

from src import models
from src.metrics import best_threshold, evaluate


def test_evaluate_counts_and_scores():
    y = [0, 0, 1, 1]
    prob = [0.1, 0.6, 0.4, 0.9]
    s = evaluate(y, prob, threshold=0.5)
    assert (s["tn"], s["fp"], s["fn"], s["tp"]) == (1, 1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["accuracy"] == 0.5


def test_evaluate_flags_nobody_without_error():
    s = evaluate([0, 1, 0], [0.1, 0.2, 0.3], threshold=0.9)
    assert s["tp"] == 0 and s["precision"] == 0 and s["recall"] == 0


def test_best_threshold_separates_perfect_scores():
    y = np.array([0] * 10 + [1] * 10)
    prob = np.r_[np.full(10, 0.2), np.full(10, 0.3)]
    t = best_threshold(y, prob)
    assert 0.2 < t <= 0.3
    assert evaluate(y, prob, t)["f1"] == 1.0


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Write tiny SGCC- and OEDI-shaped files and point the code at them."""
    rng = np.random.default_rng(0)

    def sgcc(n):
        df = pd.DataFrame(rng.random((n, 6)), columns=[f"2014-01-0{i}" for i in range(1, 7)])
        df.insert(0, "FLAG", (rng.random(n) < 0.2).astype(int))
        df.insert(0, "CONS_NO", [f"C{i}" for i in range(n)])
        return df

    def oedi(recordings):
        n = 30 * len(recordings)
        return pd.DataFrame({
            "building_type": "office",
            "recording_id": np.repeat(recordings, 30),
            "hour": np.tile(range(30), len(recordings)),
            "elec_facility_kw": rng.random(n),
            "theft_type": "Normal",
            "theft": (rng.random(n) < 0.4).astype(int),
        })

    for name, train, test in [("sgcc", sgcc(120), sgcc(40)),
                              ("oedi", oedi([1, 2, 3, 4, 5, 6]), oedi([7, 8]))]:
        (tmp_path / "data" / name).mkdir(parents=True)
        train.to_csv(tmp_path / "data" / name / "train.csv", index=False)
        test.to_csv(tmp_path / "data" / name / "test.csv", index=False)

    monkeypatch.setattr(models, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(models, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(models, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def test_load_xy_leaves_out_ids_and_answers(fake_project):
    X, y, groups = models.load_xy("oedi", "train", fake_project / "data")
    assert list(X.columns) == ["hour", "elec_facility_kw"]
    assert (X.dtypes == "float32").all()
    assert groups is not None and len(y) == 180

    X, _, groups = models.load_xy("sgcc", "train", fake_project / "data")
    assert "CONS_NO" not in X and "FLAG" not in X and groups is None


def test_oedi_folds_keep_each_recording_together(fake_project):
    _, y, groups = models.load_xy("oedi", "train", fake_project / "data")
    for train_idx, val_idx in models.make_folds(y, groups, seed=42):
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])


@pytest.mark.parametrize("dataset", ["sgcc", "oedi"])
def test_train_then_test_runs_end_to_end(fake_project, dataset):
    cv = models.train_baselines(dataset, seed=42, data_dir=fake_project / "data")
    assert list(cv["model"]) == ["Logistic Regression", "Decision Tree", "Random Forest"]
    assert cv["f1"].between(0, 1).all()
    assert (fake_project / "results" / f"baseline_cv_{dataset}_raw.csv").exists()

    test = models.test_baselines(dataset, data_dir=fake_project / "data")
    assert len(test) == 3
    assert (fake_project / "figures" / f"cm_{dataset}_raw_random_forest.png").exists()


def test_same_seed_gives_same_results(fake_project):
    a = models.train_baselines("sgcc", seed=42, data_dir=fake_project / "data")
    b = models.train_baselines("sgcc", seed=42, data_dir=fake_project / "data")
    pd.testing.assert_frame_equal(a.drop(columns="seconds"), b.drop(columns="seconds"))
