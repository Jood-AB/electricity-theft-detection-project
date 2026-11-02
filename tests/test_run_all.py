"""Tests for src/run_all.py: the steps run in the right order and the test
set is only used by the --test steps."""

from src import run_all


def modules(steps):
    """The module name of each step, e.g. 'src.models'."""
    return [args[0] for _, args in steps]


def test_training_steps_never_use_the_test_set():
    for _, args in run_all.TRAIN_STEPS:
        assert "--test" not in args


def test_every_test_step_uses_the_test_flag():
    for _, args in run_all.TEST_STEPS:
        assert "--test" in args


def test_training_steps_run_in_a_working_order():
    order = modules(run_all.TRAIN_STEPS)
    # Cleaning creates the train/test files that every other step reads
    assert order[0] == "src.prepare_data"
    # Error analysis loads the models saved by tuning
    assert order.index("src.tuning") < order.index("src.error_analysis")


def test_both_feature_sets_are_trained_and_tested():
    # Raw and with-features baselines are both needed for the comparison
    for steps in (run_all.TRAIN_STEPS, run_all.TEST_STEPS):
        model_steps = [args for _, args in steps if args[0] == "src.models"]
        assert any("features" in args for args in model_steps)
        assert any("features" not in args for args in model_steps)


def test_dry_run_does_not_start_any_step(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(run_all.subprocess, "run", lambda *a, **k: started.append(a))
    run_all.run_steps(run_all.TRAIN_STEPS, dry_run=True)
    assert started == []
    assert "python -m src.prepare_data" in capsys.readouterr().out
