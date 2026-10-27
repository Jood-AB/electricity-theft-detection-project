"""Shared scoring functions, so every member measures models the same way.

All functions take the true labels (0 = normal, 1 = theft) and the model's
predicted theft probabilities (numbers from 0 to 1).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Thresholds tried when choosing one: 0.05, 0.06, ..., 0.95
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def evaluate(y_true, prob, threshold=0.5):
    """Return all project metrics for one model as a dictionary.

    A customer (or hour) is flagged as theft when its probability is at or
    above the threshold. PR-AUC uses the probabilities directly, so it does
    not depend on the threshold.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y_true, prob, thresholds=THRESHOLDS):
    """Return the threshold with the highest F1.

    Only call this on training data (for example cross-validation
    predictions), never on the test set.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0)
              for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def plot_confusion_matrix(y_true, prob, threshold, title, path=None):
    """Draw the confusion matrix and save it to `path` if given."""
    import matplotlib
    matplotlib.use("Agg")  # draw without a screen, so it also works in scripts
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    pred = (np.asarray(prob) >= threshold).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, pred, labels=[0, 1], display_labels=["Normal", "Theft"],
        cmap="Blues", colorbar=False)
    disp.ax_.set_title(f"{title} (threshold {threshold:.2f})")
    if path is not None:
        disp.figure_.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(disp.figure_)
    return disp
