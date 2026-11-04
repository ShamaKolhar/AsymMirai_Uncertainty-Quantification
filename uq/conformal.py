"""Split conformal prediction for the binary 'cancer within h years' task.

Gives a distribution-free guarantee: on exchangeable data, the prediction set contains the
true label with probability >= 1 - alpha. Sets are {0}, {1} or {0,1}; {0,1} = "the model
can't decide at this confidence -> refer to a radiologist". This replaces a hand-picked
variance threshold.

Class-conditional (Mondrian) conformal is the default because cancer prevalence is ~1-2%:
marginal conformal could reach 90% coverage while missing most cancers.
"""
from __future__ import annotations

import numpy as np


def _qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def fit(p_calib, y_calib, alpha: float = 0.1, mondrian: bool = True) -> dict:
    p, y = np.asarray(p_calib, float), np.asarray(y_calib, int)
    s = np.where(y == 1, 1 - p, p)  # nonconformity = 1 - prob of the true class
    if mondrian:
        return {"alpha": alpha, "mondrian": True, "q0": _qhat(s[y == 0], alpha), "q1": _qhat(s[y == 1], alpha),
                "n0": int((y == 0).sum()), "n1": int((y == 1).sum())}
    q = _qhat(s, alpha)
    return {"alpha": alpha, "mondrian": False, "q0": q, "q1": q, "n0": int((y == 0).sum()), "n1": int((y == 1).sum())}


def predict_sets(p, model: dict) -> np.ndarray:
    """Returns (n, 2) bool array: column 0 = label 0 in set, column 1 = label 1 in set."""
    p = np.asarray(p, float)
    return np.stack([p <= model["q0"], (1 - p) <= model["q1"]], axis=1)


def evaluate(p_test, y_test, model: dict) -> dict:
    sets = predict_sets(p_test, model)
    y = np.asarray(y_test, int)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    return {"coverage": float(covered.mean()),
            "coverage_cancer": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
            "coverage_no_cancer": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
            "frac_set_{0}": float((sets[:, 0] & ~sets[:, 1]).mean()),
            "frac_set_{1}": float((~sets[:, 0] & sets[:, 1]).mean()),
            "frac_set_{0,1}_refer": float((size == 2).mean()),
            "frac_empty": float((size == 0).mean()),
            "target_coverage": 1 - model["alpha"]}
