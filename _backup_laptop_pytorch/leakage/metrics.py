"""Metrics for the attribute readout.

Leakage score = chance-adjusted balanced accuracy:
    L = (BA - 1/K) / (1 - 1/K)
0 means chance level for any class balance, 1 means perfect. It can be negative.
If you use a different definition in the paper, change `leakage_score` only.

Two threshold-robust variants are reported as well, because an aux-only student often has a
shifted bias on its real head, so argmax predicts one class while the ranking is informative:
  gini           = 2 * AUC - 1 (threshold-free, 0 = chance)
  leakage_prior  = leakage score after setting the binary threshold so that the predicted
                   positive rate equals the class prior (an attacker can know base rates from
                   public statistics, but no labels of the victims)
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)


def leakage_score(balanced_acc, n_classes):
    chance = 1.0 / n_classes
    return (balanced_acc - chance) / (1.0 - chance)


def classification_metrics(y_true, probs, n_classes, prior=None):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=-1)
    ba = balanced_accuracy_score(y_true, y_pred)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": ba,
        "leakage": leakage_score(ba, n_classes),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if n_classes == 2:
            out["auc"] = roc_auc_score(y_true, probs[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                       labels=list(range(n_classes)))
    except ValueError:
        out["auc"] = float("nan")
    out["gini"] = 2 * out["auc"] - 1
    out["leakage_prior"] = float("nan")
    if n_classes == 2 and prior is not None and 0 < prior < 1:
        thr = np.quantile(probs[:, 1], 1 - prior)
        y_prior = (probs[:, 1] > thr).astype(int)
        out["leakage_prior"] = leakage_score(balanced_accuracy_score(y_true, y_prior), 2)
    return {k: float(v) for k, v in out.items()}


def chance_baselines(y_true, n_classes):
    y = np.asarray(y_true).astype(int)
    p = np.bincount(y, minlength=n_classes) / len(y)
    return {"majority_acc": float(p.max()), "prior_random_acc": float((p ** 2).sum()),
            "chance_balanced_acc": 1.0 / n_classes}
